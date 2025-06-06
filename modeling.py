"""
Super Gaussian O'Connell Modeling Package 

A comprehensive package for modeling O'Connell effect EBs light curves using super-Gaussian functions
and Fourier series. This package provides tools for fitting stellar photometry
data with interactive parameter tuning capabilities.

Main Features:
- Super-Gaussian function modeling
- Fourier series fitting
- Interactive parameter tuning
- Monte Carlo integration
- O'Connell Effect Ratio (OER) and Light Curve Asymmetry (LCA) calculations

Author: David Flores
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.table import Table
from astropy.io import ascii
from astropy.stats import sigma_clip
from scipy.signal import find_peaks
from scipy import spatial
from lmfit import Model, Parameters
from symfit import parameters, variables, sin, cos, Fit
from functools import reduce
import operator
from matplotlib.widgets import Slider, Button


def super_gauss(x, b0, b1, mu, sig, p):
    """
    Super-Gaussian function for modeling stellar photometry features.
    
    The super-Gaussian is a generalization of the Gaussian function that allows
    for different shape parameters, making it suitable for modeling various
    stellar photometry features like eclipses and transits.
    
    Parameters
    ----------
    x : array-like
        Independent variable (typically phase or time).
    b0 : float
        Baseline level.
    b1 : float
        Amplitude of the feature.
    mu : float
        Center position of the feature.
    sig : float
        Width parameter (analogous to standard deviation).
    p : float
        Shape parameter (p=1 gives standard Gaussian).
        
    Returns
    -------
    numpy.ndarray
        Super-Gaussian function values.
        
    Notes
    -----
    The function form is: b0 - b1 * exp(-(((x - mu)^2) / (2 * sig^2))^p)
    """
    # Vectorized computation for better performance
    normalized_distance = (x - mu) / sig
    return b0 - b1 * np.exp(-((normalized_distance ** 2) / 2) ** p)


def fourier_series(x, f, n=0):
    """
    Generate a Fourier series model using symfit parameters.
    
    Creates a Fourier series with n harmonics for fitting periodic data.
    This is primarily used internally for parameter setup.
    
    Parameters
    ----------
    x : array-like
        Independent variable.
    f : symfit parameter
        Frequency parameter.
    n : int, optional
        Number of harmonics to include (default: 0).
        
    Returns
    -------
    symfit expression
        Fourier series expression with symbolic parameters.
    """
    a0, *cos_a = parameters(','.join([f'a{i}' for i in range(n + 1)]))
    sin_b = parameters(','.join([f'b{i}' for i in range(1, n + 1)]))
    
    series = a0
    for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1):
        series += ai * cos(i * 2 * np.pi * x) + bi * sin(i * 2 * np.pi * x)
    
    return series


def fourier_series2(x, a0, a1, a2, a3, b1, b2, b3):
    """
    Optimized Fourier series with fixed number of harmonics.
    
    This function provides a fast implementation of a 3-harmonic Fourier series
    commonly used in stellar photometry modeling.
    
    Parameters
    ----------
    x : array-like
        Independent variable (phase).
    a0 : float
        DC component (average value).
    a1, a2, a3 : float
        Cosine coefficients for harmonics 1, 2, 3.
    b1, b2, b3 : float
        Sine coefficients for harmonics 1, 2, 3.
        
    Returns
    -------
    numpy.ndarray
        Fourier series values.
        
    Notes
    -----
    This is optimized for performance by pre-computing trigonometric values
    and avoiding loops.
    """
    # Pre-compute common terms for better performance
    two_pi_x = 2 * np.pi * x
    four_pi_x = 2 * two_pi_x
    six_pi_x = 3 * two_pi_x
    
    return (a0 + 
            a1 * np.cos(two_pi_x) + a2 * np.cos(four_pi_x) + a3 * np.cos(six_pi_x) +
            b1 * np.sin(two_pi_x) + b2 * np.sin(four_pi_x) + b3 * np.sin(six_pi_x))


def find_nearest(array, value):
    """
    Find the nearest value in an array and return its index and value.
    
    Parameters
    ----------
    array : array-like
        Array to search in.
    value : float
        Value to find the nearest match for.
        
    Returns
    -------
    tuple
        (index, nearest_value) where index is the position and 
        nearest_value is the actual closest value.
        
    Examples
    --------
    >>> arr = np.array([1, 3, 5, 7, 9])
    >>> idx, val = find_nearest(arr, 6)
    >>> print(f"Index: {idx}, Value: {val}")
    Index: 2, Value: 5
    """
    array = np.asarray(array)
    idx = np.abs(array - value).argmin()
    return idx, array[idx]


def gg_fit9_factory(params):
    """
    Factory function to create a composite model from fitted parameters.
    
    Creates a function that combines Fourier series and three super-Gaussian
    components based on the fitted parameters from the modeling process.
    
    Parameters
    ----------
    params : list
        List of 22 parameters:
        - params[0:7]: Fourier series coefficients
        - params[7:12]: First super-Gaussian parameters (c component)
        - params[12:17]: Second super-Gaussian parameters (f component)
        - params[17:22]: Third super-Gaussian parameters (v component)
        
    Returns
    -------
    function
        Composite model function that takes phase as input and returns
        the combined model evaluation.
        
    Notes
    -----
    The returned function combines:
    - Fourier series for periodic variations
    - Three super-Gaussian components for eclipse/transit features
    """
    def gg_fit9(o):
        """Composite model evaluation function."""
        # Extract parameter groups for better readability
        fourier_params = params[0:7]
        c_params = params[7:12]
        f_params = params[12:17] 
        v_params = params[17:22]
        
        # Compute components
        g = fourier_series2(o, *fourier_params)
        c = super_gauss(o, *c_params)
        f = super_gauss(o, *f_params)
        v = super_gauss(o, *v_params)
        
        return g + c + f + v
    
    return gg_fit9


def integrate_mcmc(func, xmin, xmax, N=1000000):
    """
    Monte Carlo integration with error estimation.
    
    Performs Monte Carlo integration of a function over a specified interval
    and provides an estimate of the integration error.
    
    Parameters
    ----------
    func : callable
        Function to integrate. Must accept array input.
    xmin, xmax : float
        Integration limits.
    N : int, optional
        Number of Monte Carlo samples (default: 1,000,000).
        
    Returns
    -------
    tuple
        (integral_value, error_estimate) where both are floats.
        
    Notes
    -----
    The error estimate assumes the function values are normally distributed
    and uses the standard error of the mean formula.
    
    Examples
    --------
    >>> # Integrate x^2 from 0 to 1
    >>> integral, error = integrate_mcmc(lambda x: x**2, 0, 1, N=100000)
    >>> print(f"Integral: {integral:.4f} ± {error:.4f}")
    """
    # Generate random samples uniformly distributed in [xmin, xmax]
    randx = np.random.uniform(xmin, xmax, N)
    y = func(randx)
    
    # Monte Carlo integration formula
    integral = (xmax - xmin) * np.mean(y)
    
    # Error estimation using standard error of the mean
    variance = np.mean(y**2) - np.mean(y)**2
    error = (xmax - xmin) * np.sqrt(variance / N)
    
    return integral, error


def get_gg_fit9_expression_from_params(params):
    """
    Generate human-readable mathematical expression from fitted parameters.
    
    Converts the numerical parameters from the composite model into a
    formatted mathematical expression that can be used for documentation
    or publication purposes.
    
    Parameters
    ----------
    params : list
        List of 22 fitted parameters from the gg_fit9 model.
        
    Returns
    -------
    str
        Formatted mathematical expression showing the complete model.
        
    Examples
    --------
    >>> params = [1.0, 0.1, 0.05, 0.02, 0.08, 0.03, 0.01,  # Fourier
    ...           1.0, 0.2, 0.5, 0.1, 1.5,                 # Super-Gauss 1
    ...           1.0, 0.3, 0.25, 0.08, 1.2,               # Super-Gauss 2  
    ...           1.0, 0.15, 0.75, 0.12, 1.8]              # Super-Gauss 3
    >>> expr = get_gg_fit9_expression_from_params(params)
    >>> print(expr)
    """
    # Extract parameter groups
    fourier_params = params[0:7]
    sg1_params = params[7:12]   # component c
    sg2_params = params[12:17]  # component f
    sg3_params = params[17:22]  # component v
    
    a0, a1, a2, a3, b1, b2, b3 = fourier_params

    def format_fourier():
        """Format Fourier series component."""
        terms = [f"{a0:.5f}"]
        
        cos_terms = [f"{a1:.5f}*cos(2πx)", f"{a2:.5f}*cos(4πx)", f"{a3:.5f}*cos(6πx)"]
        sin_terms = [f"{b1:.5f}*sin(2πx)", f"{b2:.5f}*sin(4πx)", f"{b3:.5f}*sin(6πx)"]
        
        terms.extend(cos_terms + sin_terms)
        return " + ".join(terms)

    def format_super_gauss(b0, b1, mu, sig, p):
        """Format super-Gaussian component."""
        return (f"{b0:.5f} - {b1:.5f} * exp(-"
                f"(((x - {mu:.5f})^2) / (2 * {sig:.5f}^2))^{p:.5f})")

    # Build complete expression
    fourier_expr = format_fourier()
    sg1_expr = format_super_gauss(*sg1_params)
    sg2_expr = format_super_gauss(*sg2_params)
    sg3_expr = format_super_gauss(*sg3_params)
    
    expression = (
        "sG_Model(x) =\n"
        f"{fourier_expr}\n +\n"
        f"{sg1_expr}\n +\n"
        f"{sg2_expr}\n +\n"
        f"{sg3_expr}"
    )
    
    return expression


def interactive_super_gauss_tuner(x, y, init_params, title_suffix=""):
    """
    Interactive parameter tuning tool for super-Gaussian functions.
    
    Provides a graphical interface with sliders for real-time adjustment
    of super-Gaussian parameters. This tool is essential for getting good
    initial parameter estimates before automated fitting.
    
    Parameters
    ----------
    x : array-like
        Independent variable data (phase or time).
    y : array-like
        Dependent variable data (magnitude or flux).
    init_params : tuple
        Initial parameter guess as (b0, b1, mu, sig, p).
    title_suffix : str, optional
        Additional string to append to the plot title (default: "").
        
    Returns
    -------
    dict
        Dictionary containing the final selected parameters:
        {'b0': float, 'b1': float, 'mu': float, 'sig': float, 'p': float}
        
    Notes
    -----
    This function opens an interactive matplotlib window. The user should:
    1. Adjust sliders to match the data visually
    2. Click "Confirm" to accept the parameters
    3. The window will close and parameters will be returned
    
    The sliders have the following ranges:
    - b0: [min(y)-1, max(y)+1] (baseline level)
    - b1: [0, 2*max(|y|)] (amplitude)
    - mu: [min(x), max(x)] (center position)  
    - sig: [0.001, 1.0] (width)
    - p: [0.5, 5.0] (shape parameter)
    
    Examples
    --------
    >>> phase = np.linspace(0, 1, 100)
    >>> flux = 1.0 - 0.3 * np.exp(-((phase - 0.5)/0.1)**2)
    >>> params = interactive_super_gauss_tuner(phase, flux, 
    ...                                       (1.0, 0.3, 0.5, 0.1, 1.0),
    ...                                       "Star HD 12345")
    >>> print("Selected parameters:", params)
    """
    b0_init, b1_init, mu_init, sig_init, p_init = init_params
    final_params = {}

    # Set up the main plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.25, bottom=0.45)
    
    # Plot data and initial model
    l_data, = ax.plot(x, y, 'o', label='Data', alpha=0.7, markersize=4)
    l_model, = ax.plot(x, super_gauss(x, *init_params), 'r-', 
                      label='Super-Gauss', linewidth=2)
    
    # Create base title and add suffix if provided
    base_title = 'Initial Super-Gaussian Parameter Tuning'
    if title_suffix:
        full_title = f'{base_title} ({title_suffix})'
    else:
        full_title = base_title
    
    ax.set_title(full_title, fontsize=14)
    ax.set_xlabel('Phase', fontsize=12)
    ax.set_ylabel('Normalized Magnitude/Flux', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Create slider axes with better spacing
    axcolor = 'lightgoldenrodyellow'
    slider_height = 0.03
    slider_spacing = 0.04
    slider_left = 0.25
    slider_width = 0.65
    
    ax_b0 = plt.axes([slider_left, 0.36, slider_width, slider_height], facecolor=axcolor)
    ax_b1 = plt.axes([slider_left, 0.36-slider_spacing, slider_width, slider_height], facecolor=axcolor)
    ax_mu = plt.axes([slider_left, 0.36-2*slider_spacing, slider_width, slider_height], facecolor=axcolor)
    ax_sig = plt.axes([slider_left, 0.36-3*slider_spacing, slider_width, slider_height], facecolor=axcolor)
    ax_p = plt.axes([slider_left, 0.36-4*slider_spacing, slider_width, slider_height], facecolor=axcolor)

    # Create sliders with improved ranges
    y_range = max(y) - min(y)
    s_b0 = Slider(ax_b0, 'b0 (baseline)', min(y)-y_range*0.2, max(y)+y_range*0.2, 
                  valinit=b0_init, valfmt='%.3f')
    s_b1 = Slider(ax_b1, 'b1 (amplitude)', 0, y_range*2, 
                  valinit=b1_init, valfmt='%.3f')
    s_mu = Slider(ax_mu, 'mu (center)', min(x), max(x), 
                  valinit=mu_init, valfmt='%.3f')
    s_sig = Slider(ax_sig, 'sig (width)', 0.001, (max(x)-min(x))*0.5, 
                   valinit=sig_init, valfmt='%.4f')
    s_p = Slider(ax_p, 'p (shape)', 0.5, 5.0, 
                 valinit=p_init, valfmt='%.2f')

    def update(val):
        """Update the model plot when sliders change."""
        params = (s_b0.val, s_b1.val, s_mu.val, s_sig.val, s_p.val)
        y_fit = super_gauss(x, *params)
        l_model.set_ydata(y_fit)
        
        # Calculate and display R-squared
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Update title with R-squared and suffix
        r_squared_title = f'Interactive Super-Gaussian Tuning (R² = {r_squared:.4f})'
        if title_suffix:
            updated_title = f'{r_squared_title} - {title_suffix}'
        else:
            updated_title = r_squared_title
            
        ax.set_title(updated_title, fontsize=14)
        
        fig.canvas.draw_idle()

    # Connect sliders to update function
    for slider in [s_b0, s_b1, s_mu, s_sig, s_p]:
        slider.on_changed(update)

    # Add confirm button
    confirm_ax = plt.axes([0.4, 0.08, 0.2, 0.05])
    button = Button(confirm_ax, 'Confirm Parameters', 
                   color=axcolor, hovercolor='lightgreen')

    def on_confirm(event):
        """Handle parameter confirmation."""
        final_params.update({
            'b0': s_b0.val,
            'b1': s_b1.val, 
            'mu': s_mu.val,
            'sig': s_sig.val,
            'p': s_p.val
        })
        
        print("\n" + "="*50)
        print("CONFIRMED SUPER-GAUSSIAN PARAMETERS:")
        if title_suffix:
            print(f"FOR: {title_suffix}")
        print("="*50)
        for param, value in final_params.items():
            print(f"{param:>3} = {value:>10.5f}")
        print("="*50)
        
        plt.close(fig)

    button.on_clicked(on_confirm)
    
    # Add instructions
    fig.text(0.02, 0.5, 
             "Instructions:\n"
             "• Adjust sliders to fit the data\n"
             "• R² value shows goodness of fit\n"
             "• Click 'Confirm' when satisfied\n"
             "• Window will close automatically", 
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.show()
    return final_params

def sp_modeling(phase, mag, error):
    """
    Comprehensive stellar photometry modeling function.
    
    This is the main function for modeling stellar light curves using a combination
    of Fourier series and super-Gaussian functions. It handles eclipsing binaries,
    transiting exoplanets, and other variable star phenomena.
    
    The function performs:
    1. Interactive parameter tuning for super-Gaussian components
    2. Automated fitting of composite model (Fourier + 3 super-Gaussians)
    3. Calculation of astrophysical parameters (OER, LCA)
    4. Generation of diagnostic plots
    
    Parameters
    ----------
    phase : array-like
        Orbital phase values (typically 0 to 1).
    mag : array-like  
        Magnitude or normalized flux measurements.
    error : array-like
        Measurement uncertainties (same length as phase and mag).
        
    Returns
    -------
    None
        Function prints results and displays plots. Future versions may
        return fitted parameters and derived quantities.
        
    Notes
    -----
    **Model Components:**
    
    The fitted model consists of:
    - Fourier series (3 harmonics): captures periodic variations
    - Super-Gaussian 1 (g_l): primary eclipse/transit  
    - Super-Gaussian 2 (g_l2): secondary eclipse/transit
    - Super-Gaussian 3 (g_4): additional features (spots, etc.)
    
    **Derived Parameters:**
    
    - OER (Orbital Eccentricity Ratio): Ratio of eclipse depths, related to
      orbital eccentricity and stellar temperature ratio
    - LCA (Light Curve Asymmetry): Quantifies asymmetry in the light curve,
      sensitive to stellar activity and tidal distortions
      
    **Interactive Usage:**
    
    The function will open interactive windows for parameter tuning. For each
    super-Gaussian component:
    1. Adjust sliders to visually match the data feature
    2. Click "Confirm Parameters" to proceed
    3. Repeat for next component
    
    **Output:**
    
    - Mathematical expression of fitted model
    - OER and LCA values with uncertainties  
    - Multi-panel diagnostic plot showing:
      * Individual model components
      * Final combined model vs data
      
    Examples
    --------
    >>> # Load your light curve data
    >>> phase, magnitude, errors = load_lightcurve_data()
    >>> 
    >>> # Ensure phase is folded to [0, 1] range
    >>> phase = phase % 1.0
    >>> 
    >>> # Run the modeling (interactive)
    >>> sp_modeling(phase, magnitude, errors)
    
    **Tips for Best Results:**
    
    - Ensure data is properly phase-folded
    - Remove obvious outliers before modeling
    - Start with reasonable initial guesses in interactive tuning
    - Primary eclipse should be near phase = 0 or 1
    - Secondary eclipse should be near phase = 0.5
    """
    
    print("\n" + "="*60)
    print("STELLAR PHOTOMETRY MODELING")  
    print("="*60)
    print(f"Data points: {len(phase)}")
    print(f"Phase range: [{np.min(phase):.3f}, {np.max(phase):.3f}]")
    print(f"Magnitude range: [{np.min(mag):.3f}, {np.max(mag):.3f}]")
    print("="*60)

    # Sort data by phase for consistent processing
    sorted_indices = np.argsort(phase)
    phase, mag, error = phase[sorted_indices], mag[sorted_indices], error[sorted_indices]

    # Create extended phase coverage for better edge handling
    phase2 = np.concatenate([phase - 1, phase, phase + 1])
    mag2 = np.tile(mag, 3)
    error2 = np.tile(error, 3)
    
    # Define phase masks for different fitting regions
    mask = (0.8 <= phase2) & (phase2 <= 1.2)      # Primary eclipse region
    mask_ext = (-0.2 <= phase2) & (phase2 <= 1.2)  # Extended fitting range

    print(f"\nFitting primary eclipse (phase 0.8-1.2)...")
    print("Please use the interactive tuner for the PRIMARY ECLIPSE feature.")
    
    # Primary eclipse modeling (around phase = 1.0)
    gauss_l = Model(super_gauss, prefix='g_l')
    fo = interactive_super_gauss_tuner(
        phase2[mask], mag2[mask], 
        init_params=(np.median(mag2[mask]), 
                    abs(np.min(mag2[mask]) - np.max(mag2[mask])), 
                    1.0, 0.1, 1.0),
        title_suffix='sG-1')
    
    # Fit primary eclipse with interactive parameters
    params_gaussl = gauss_l.make_params(
        b0=fo['b0'], b1=fo['b1'], sig=fo['sig'], mu=fo['mu'], p=fo['p']
    )
    g_l = gauss_l.fit(mag2[mask], params_gaussl, x=phase2[mask], 
                     weights=np.sqrt(1.0 / error2[mask]))
    par_l = [g_l.params[k].value for k in g_l.params]

    # Set up parameters for shifted primary eclipse (g_l2)
    params_gaussl = Parameters()
    params_gaussl.add_many(
        ('g_lb0', par_l[0]), 
        ('g_lb1', par_l[1], False),      # Fixed amplitude
        ('g_lmu', par_l[2] - 1, False),  # Shifted center
        ('g_lsig', par_l[3], False),     # Fixed width
        ('g_lp', par_l[4], False)        # Fixed shape
    )

    # Second primary eclipse component  
    gauss_l2 = Model(super_gauss, prefix='g_l2')
    params_gaussl2 = Parameters()
    params_gaussl2.add_many(
        ('g_l2b0', par_l[0]), 
        ('g_l2b1', par_l[1], False),
        ('g_l2mu', par_l[2], False), 
        ('g_l2sig', par_l[3], False),
        ('g_l2p', par_l[4], False)
    )

    # Fourier series fitting for periodic variations
    print("\nFitting Fourier series for periodic variations...")
    fourier = Model(fourier_series2, prefix='fourier')
    x, y = variables('x, y')
    w, = parameters('w')
    
    # Use symfit for initial Fourier parameter estimation
    mod_fit = Fit({y: fourier_series(x, f=w, n=3)}, x=phase, y=mag).execute()
    params_fourier = fourier.make_params(
        **{name: (param.value if hasattr(param, 'value') else param)
           for name, param in mod_fit.params.items()}
    )

    # Secondary eclipse modeling (around phase = 0.5)
    print(f"\nFitting secondary eclipse/feature (phase 0.2-0.8)...")
    print("Please use the interactive tuner for the SECONDARY ECLIPSE feature.")
    
    mask_g3 = (0.2 <= phase) & (phase <= 0.8)
    gauss_3 = Model(super_gauss, prefix='g_3')
    
    if np.any(mask_g3):
        fo = interactive_super_gauss_tuner(
            phase[mask_g3], mag[mask_g3], 
            init_params=(np.median(mag[mask_g3]),
                        abs(np.min(mag[mask_g3]) - np.max(mag[mask_g3])), 
                        0.5, 0.1, 1.0),
            title_suffix='sG-2'
        )
        
        params_gauss3 = gauss_3.make_params(
            b0=fo['b0'], b1=fo['b1'], sig=fo['sig'], mu=fo['mu'], p=fo['p']
        )
        g_3 = gauss_3.fit(mag[mask_g3], params_gauss3, x=phase[mask_g3], 
                         weights=np.sqrt(1.0 / error[mask_g3]))
    else:
        # Fallback if no secondary eclipse region
        print("No data in secondary eclipse region, using default parameters.")
        g_3 = gauss_3.fit(mag, gauss_3.make_params(
            b0=np.median(mag), b1=0.1, mu=0.5, sig=0.1, p=1.0
        ), x=phase)

    # Set up parameters for final secondary eclipse component
    params_gauss4 = Parameters()
    for name in ['b0', 'b1', 'mu', 'sig', 'p']:
        vary = name not in ['b1', 'mu']  # Fix amplitude and center
        params_gauss4.add(f'g_4{name}', 
                         value=g_3.params[f'g_3{name}'].value, vary=vary)
    
    # Constraint on shape parameter
    if g_3.params['g_3p'].value < 0.79:
        params_gauss4['g_4p'].min = 0.85

    # Combine all models for final fit
    print("\nPerforming final composite model fit...")
    custom_model = fourier + Model(super_gauss, prefix='g_4') + gauss_l + gauss_l2
    all_params = params_fourier + params_gauss4 + params_gaussl + params_gaussl2
    
    # Final fit with all components
    result = custom_model.fit(mag2[mask_ext], all_params, x=phase2[mask_ext], 
                             weights=np.sqrt(1.0 / error2[mask_ext]))
    
    # Extract fitted parameters and create composite model
    param_values = [result.params[p].value for p in result.params]
    comps = result.eval_components()
    gg_fit9 = gg_fit9_factory(param_values)
    
    # Generate and display mathematical expression
    expression = get_gg_fit9_expression_from_params(param_values)
    print(f"\n{expression}")

    # Calculate derived astrophysical parameters
    print("\nCalculating astrophysical parameters...")
    
    # Orbital Eccentricity Ratio (OER)
    I1, err1 = integrate_mcmc(
        lambda t: gg_fit9(t) - np.min(gg_fit9(phase)), 0, 0.5, N=500000
    )
    I2, err2 = integrate_mcmc(
        lambda t: gg_fit9(t) - np.min(gg_fit9(phase)), 0.5, 1, N=500000
    )
    
    # Light Curve Asymmetry (LCA) 
    I3, err3 = integrate_mcmc(
        lambda t: ((gg_fit9(t) - gg_fit9(1 - t)) ** 2) / (gg_fit9(t) ** 2), 
        0, 0.5, N=500000
    )

    # Calculate final parameters with error propagation
    OER = I1 / I2 if I2 != 0 else np.inf
    OER_err = (I1 / I2) * np.sqrt((err1 / I1) ** 2 + (err2 / I2) ** 2) if I2 != 0 and I1 != 0 else 0
    
    LCA = np.sqrt(I3) if I3 >= 0 else 0
    LCA_err = 0.5 * err3 / np.sqrt(I3) if I3 > 0 else 0

    # Display results
    print("\n" + "="*50)
    print("ASTROPHYSICAL PARAMETERS")
    print("="*50)
    print(f"OER (O'Connell Effect Ratio): {OER:.5f} ± {OER_err:.5f}")
    print(f"LCA (Light Curve Asymmetry): {LCA:.5f} ± {LCA_err:.5f}")
    print("="*50)

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    _create_diagnostic_plots(phase, mag, error, phase2, mask_ext, comps, gg_fit9, result)
    
    print("\nModeling completed successfully!")
    print("="*60)


def _create_diagnostic_plots(phase, mag, error, phase2, mask_ext, comps, gg_fit9, result):
    """
    Create comprehensive diagnostic plots for the fitted model.
    
    Parameters
    ----------
    phase : array-like
        Original phase data.
    mag : array-like
        Original magnitude data.
    error : array-like
        Original error data.
    phase2 : array-like
        Extended phase data.
    mask_ext : array-like
        Extended mask for plotting.
    comps : dict
        Model components from lmfit.
    gg_fit9 : callable
        Composite model function.
    result : lmfit.ModelResult
        Fitting result object.
    """
    
    def plot_component(ax, x, y, label, legend_loc=0, color='k'):
        """Helper function to create subplot with common formatting."""
        ax.plot(x, y, '--', label=label, color=color, linewidth=1.5)
        ax.legend(loc=legend_loc, fontsize=10)
        ax.tick_params(bottom=True, top=True, left=True, right=True, 
                      direction='in', which='both', labelbottom=True, 
                      labeltop=False, labelleft=True, labelright=False)
        ax.minorticks_on()
        #ax.grid(True, alpha=0.3)
        
        # Reduce y-tick density for cleaner appearance
        yticks = ax.get_yticks()
        if len(yticks) > 6:
            ax.set_yticks(yticks[::2])

    # Create figure with improved layout
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 2, 0.1], hspace=0.3, wspace=0.3)

    # Component subplot definitions with colors
    components = [
        ('fourier', (0, 0), 'Fourier', 0, 'black'),
        ('g_l2',    (0, 1), 'sG-1', 0, 'black'),
        ('g_4',     (1, 0), 'sG-2', 0, 'black'),
        ('g_l',     (1, 1), 'sG-3', 0, 'black'),
    ]

    # Plot individual components
    for comp_key, pos, label, loc, color in components:
        ax = plt.subplot(gs[pos[0], pos[1]])
        if comp_key in comps:
            plot_component(ax, phase2[mask_ext], comps[comp_key], label, loc, color)

    # Combined final model plot (larger subplot)
    ax_main = plt.subplot(gs[2, :])
    
    # High-resolution model for smooth plotting
    x_model = np.linspace(0, 1, 1000)
    y_model = gg_fit9(x_model)
    
    # Plot model and data
    ax_main.plot(x_model, y_model, color='black', 
                label='Final Model', zorder=100)
    ax_main.errorbar(phase, mag, yerr=error, fmt='o', color='#009988',
                     lw=0.5, ms=2, zorder=50)
    
    # Calculate and display residuals statistics
    residuals = mag - gg_fit9(phase)
    rms_residual = np.sqrt(np.mean(residuals**2))
    
    ax_main.legend(fontsize=11, loc='upper right')
    ax_main.tick_params(bottom=True, top=True, left=True, right=True, 
                       direction='in', which='both', labelbottom=True, 
                       labeltop=False, labelleft=True, labelright=False)
    ax_main.minorticks_on()
    #ax_main.grid(True, alpha=0.3)
    ax_main.set_ylabel('Normalized Flux / Magnitude', fontsize=9)
    ax_main.set_xlabel('Phase', fontsize=12)
    
    # Add fit statistics as text box
    '''
    stats_text = f'RMS Residual: {rms_residual:.4f}\nReduced χ²: {result.redchi:.3f}'
    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.8), fontsize=10)
    '''

    # Add residuals plot in the bottom space
    '''
    ax_residual = plt.subplot(gs[3, :])
    ax_residual.errorbar(phase, residuals, yerr=error, fmt='o', 
                        color='red', alpha=0.6, markersize=2)
    ax_residual.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax_residual.axhline(y=rms_residual, color='gray', linestyle='--', alpha=0.5, 
                       label=f'±RMS ({rms_residual:.4f})')
    ax_residual.axhline(y=-rms_residual, color='gray', linestyle='--', alpha=0.5)
    ax_residual.set_ylabel('Residuals', fontsize=10)
    ax_residual.set_xlabel('Orbital Phase', fontsize=12)
    ax_residual.tick_params(direction='in', which='both')
    ax_residual.minorticks_on()
    ax_residual.grid(True, alpha=0.3)
    ax_residual.legend(fontsize=9)
    '''

    #plt.tight_layout()
    plt.show()


# Additional utility functions for enhanced functionality
def validate_input_data(phase, mag, error):
    """
    Validate input data for the modeling function.
    
    Parameters
    ----------
    phase, mag, error : array-like
        Input data arrays.
        
    Raises
    ------
    ValueError
        If data validation fails.
        
    Returns
    -------
    tuple
        Validated and converted numpy arrays.
    """
    # Convert to numpy arrays
    phase = np.asarray(phase, dtype=float)
    mag = np.asarray(mag, dtype=float)
    error = np.asarray(error, dtype=float)
    
    # Check array lengths
    if not (len(phase) == len(mag) == len(error)):
        raise ValueError("Input arrays must have the same length")
    
    # Check for minimum data points
    if len(phase) < 10:
        raise ValueError("At least 10 data points required for modeling")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(phase)) or np.any(~np.isfinite(mag)) or np.any(~np.isfinite(error)):
        raise ValueError("Input data contains NaN or infinite values")
    
    # Check for positive errors
    if np.any(error <= 0):
        raise ValueError("All error values must be positive")
    
    # Warn about phase range
    if np.max(phase) - np.min(phase) < 0.8:
        print("Warning: Phase coverage is less than 0.8. Results may be unreliable.")
    
    return phase, mag, error


def save_results(filename, params, oer, oer_err, lca, lca_err, expression):
    """
    Save modeling results to a file.
    
    Parameters
    ----------
    filename : str
        Output filename.
    params : list
        Fitted parameter values.
    oer, oer_err : float
        OER value and uncertainty.
    lca, lca_err : float
        LCA value and uncertainty.
    expression : str
        Mathematical expression of the model.
    """
    with open(filename, 'w') as f:
        f.write("Stellar Photometry Modeling Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Model Expression:\n")
        f.write(expression + "\n\n")
        f.write("Astrophysical Parameters:\n")
        f.write(f"OER = {oer:.6f} ± {oer_err:.6f}\n")
        f.write(f"LCA = {lca:.6f} ± {lca_err:.6f}\n\n")
        f.write("Fitted Parameters:\n")
        for i, param in enumerate(params):
            f.write(f"p[{i:2d}] = {param:12.6f}\n")


# Example usage and documentation
if __name__ == "__main__":
    """
    Example usage of the stellar photometry modeling package.
    
    This example demonstrates how to use the sp_modeling function
    with synthetic data.
    """
    
    # Generate synthetic light curve data
    np.random.seed(42)
    phase_example = np.linspace(0, 1, 200)
    
    # Create synthetic eclipsing binary light curve
    # Primary eclipse (deeper) at phase = 0
    primary_eclipse = 0.3 * np.exp(-((phase_example - 0.0) / 0.05) ** 2)
    primary_eclipse += 0.3 * np.exp(-((phase_example - 1.0) / 0.05) ** 2)
    
    # Secondary eclipse (shallower) at phase = 0.5  
    secondary_eclipse = 0.1 * np.exp(-((phase_example - 0.5) / 0.04) ** 2)
    
    # Add some periodic variations
    periodic_variation = 0.02 * np.sin(2 * np.pi * phase_example)
    
    # Combine components
    mag_example = 1.0 - primary_eclipse - secondary_eclipse + periodic_variation
    
    # Add realistic noise
    error_example = np.full_like(phase_example, 0.01)
    noise = np.random.normal(0, error_example)
    mag_example += noise
    
    print("Example synthetic data generated.")
    print("To run the modeling on this example data, uncomment the following line:")
    print("# sp_modeling(phase_example, mag_example, error_example)")
    
    #Uncomment the next line to test with synthetic data
    #sp_modeling(phase_example, mag_example, error_example)