import numpy as np
import h5py
import os
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import re
from scipy.signal import stft, detrend, istft, hilbert
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

def get_tidal_info(filename, station="9449828", current_station="PUG1724_31", print_output=True):
    """
    Fetch tide height + tidal current from NOAA based on filename timestamp.
    Uses a 2-day NOAA window to allow interpolation across midnight.
    Returns a clean dict.
    """

    # =========================================================
    # 1. PARSE TIME
    # =========================================================
    try:
        base = os.path.basename(filename)
        parts = base.split('_')

        date_str = parts[1]
        time_str = parts[2]

        query_dt_str = f"{date_str} {time_str.replace('.', ':')}"
        query_time = datetime.strptime(query_dt_str, "%Y-%m-%d %H:%M:%S")

        tq = query_time.timestamp()

        # ---- FIX: extend to next day for interpolation safety
        date0 = datetime.strptime(date_str, "%Y-%m-%d")
        date1 = date0 + timedelta(days=1)

        bdate = date0.strftime("%Y%m%d")
        edate = date1.strftime("%Y%m%d")

    except Exception as e:
        raise ValueError(f"Filename parse error: {e}")

    # =========================================================
    # 2. FETCH TIDE HEIGHT (Station 9449828 usually Friday Harbor)
    # =========================================================
    height = np.nan
    try:
        url_tide = (
            "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
            f"product=predictions&application=NOS.COOPS.TAC.WL&begin_date={bdate}&end_date={edate}&"
            f"datum=MLLW&station={station}&time_zone=GMT&units=metric&interval=hilo&format=json"
        )
        r_tide = requests.get(url_tide, timeout=10)
        tide_data = r_tide.json()

        if "predictions" in tide_data:
            preds = tide_data["predictions"]
            t_sec = np.array([datetime.strptime(p["t"], "%Y-%m-%d %H:%M").timestamp() for p in preds])
            h_vals = np.array([float(p["v"]) for p in preds])

            for i in range(len(t_sec) - 1):
                if t_sec[i] <= tq <= t_sec[i+1]:
                    dt = t_sec[i+1] - t_sec[i]
                    if dt == 0: break
                    A = (h_vals[i] - h_vals[i+1]) / 2
                    mid = (h_vals[i] + h_vals[i+1]) / 2
                    phase = np.pi * (tq - t_sec[i]) / dt
                    height = mid + A * np.cos(phase)
                    break
    except:
        pass

    # =========================================================
    # 3. FETCH TIDAL CURRENT (Station PUG1724_31 for Haro Strait)
    # =========================================================
    current_val = np.nan
    closest_slack = None
    closest_max = None

    try:
        # NOTE: product=predictions on a currents station returns slack/max predictions
        url_curr = (
            "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
            f"product=predictions&application=NOS.COOPS.TAC.WL&begin_date={bdate}&end_date={edate}&"
            f"station={current_station}&time_zone=GMT&units=metric&format=json"
        )
        r_curr = requests.get(url_curr, timeout=10)
        curr_data = r_curr.json()

        if "predictions" in curr_data:
            events = curr_data["predictions"]
            e_list = []
            for e in events:
                ts = datetime.strptime(e["t"], "%Y-%m-%d %H:%M").timestamp()
                e_list.append((ts, float(e["v"]), e["type"]))

            # Interpolate current at tq
            for i in range(len(e_list) - 1):
                t0, v0, type0 = e_list[i]
                t1, v1, type1 = e_list[i+1]
                if t0 <= tq <= t1:
                    dt = t1 - t0
                    if dt == 0: break
                    # Sinusoidal interpolation between slack and max
                    phase = (np.pi/2) * (tq - t0) / dt
                    if "slack" in type0.lower():
                        # From slack to max: sin increases from 0 to 1
                        current_val = v1 * np.sin(phase)
                    else:
                        # From max to slack: cos decreases from 1 to 0
                        current_val = v0 * np.cos(phase)
                    break

            # Find closest slack and max
            slacks = [e for e in e_list if "slack" in e[2].lower()]
            if slacks: closest_slack = min(slacks, key=lambda x: abs(x[0] - tq))
            maxes = [e for e in e_list if "max" in e[2].lower()]
            if maxes: closest_max = min(maxes, key=lambda x: abs(x[0] - tq))

    except Exception as e:
        if print_output:
            print(f"Error fetching current data: {e}")

    # =========================================================
    # 4. PRINT RESULTS
    # =========================================================
    if print_output:
        print(f"\n--- Tide & Current Info ({query_dt_str}) ---")
        print(f"Location:      Haro Strait (Station {current_station})")
        print(f"Tide height:   {height:.2f} m (Friday Harbor)")
        print(f"Tidal current: {current_val:.2f} m/s")

        if closest_slack:
            slack_time = datetime.fromtimestamp(closest_slack[0]).strftime("%H:%M")
            print(f"Closest slack: {slack_time} GMT")
        if closest_max:
            max_time = datetime.fromtimestamp(closest_max[0]).strftime("%H:%M")
            print(f"Closest max:   {max_time} GMT ({closest_max[1]:.2f} m/s, {closest_max[2]})")
        print("--------------------------------------------------\n")

    # =========================================================
    # 5. RETURN
    # =========================================================
    return {
        "tide_height": float(height) if not np.isnan(height) else np.nan,
        "tidal_current": float(current_val) if not np.isnan(current_val) else np.nan,
        "closest_slack": closest_slack,
        "closest_max": closest_max,
        "datetime": query_dt_str
    }
def get_weather_info(filename, location_id="5971146", print_output=True):

    try:
        base = os.path.basename(filename)
        parts = base.split('_')
        date_str = parts[1]
        time_str = parts[2]

        query_dt_str = f"{date_str} {time_str.replace('.', ':')}"
        dt = datetime.strptime(query_dt_str, "%Y-%m-%d %H:%M:%S")

        year, month, day = dt.year, dt.month, dt.day
        hd = f"{year}{month:02d}{day:02d}"

    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

    url = (
        f"https://www.timeanddate.com/scripts/cityajax.php?"
        f"n=@{location_id}&mode=historic&hd={hd}&month={month}&year={year}&json=1"
    )

    try:
        response = requests.get(url, timeout=10)
        text = response.text

        pattern = r'\{c:\[\{h:"([^"]+)"\},.*?\{h:"(-?\d+)&nbsp;°F"\},.*?\{s:"sep",h:"([^"]+)\s*mph"\}'
        matches = re.findall(pattern, text)

        if not matches:
            return None

        weather_data = []

        for time_h, temp_f, wind_mph in matches:
            t_str = time_h.split('<br>')[0]

            try:
                obs_time = datetime.strptime(f"{date_str} {t_str}", "%Y-%m-%d %I:%M %p")
            except:
                continue

            temp_c = (float(temp_f) - 32) * 5 / 9

            if "No wind" in wind_mph:
                wind_kmh = 0.0
            else:
                w_val_str = re.sub(r'&\w+;', '', wind_mph).strip()
                try:
                    wind_kmh = float(w_val_str) * 1.60934
                except:
                    wind_kmh = 0.0

            weather_data.append((obs_time, temp_c, wind_kmh))

        if not weather_data:
            return None

        closest = min(weather_data, key=lambda x: abs(x[0] - dt))
        obs_time_match, temp_c_match, wind_kmh_match = closest

    except Exception as e:
        print(f"Weather fetch error: {e}")
        return None

    if print_output:
        print(f"\n--- Weather Info ({query_dt_str}) ---")
        print(f"Closest time: {obs_time_match.strftime('%I:%M %p')}")
        print(f"Temp:         {temp_c_match:.1f} °C")
        print(f"Wind:         {wind_kmh_match:.1f} km/h")
        print("----------------------------------\n")

    return {
        "temperature": float(temp_c_match),
        "wind_speed": float(wind_kmh_match),
        "datetime": query_dt_str
    }


def get_environmental_metadata(filename):

    tide = get_tidal_info(filename, print_output=False)
    weather = get_weather_info(filename, print_output=False)

    return {
        "tide_height": tide["tide_height"] if tide else np.nan,
        "tidal_current": tide["tidal_current"] if tide else np.nan,
        "temperature": weather["temperature"] if weather else np.nan,
        "wind_speed": weather["wind_speed"] if weather else np.nan,
        "datetime": tide["datetime"] if tide else None
    }
def get_weather_at_sunset_point(target_dates):
    """
    Extracts Sea Surface Temperature and Wind Speed from ROMS NetCDF model data.
    """
    file_path = 'juber_2025.11.01_2026.01.31(1).nc'
    base_time = datetime(1970, 1, 1)

    try:
        with h5py.File(file_path, 'r') as f:
            ocean_time = f['ocean_time'][:]
            # SST is the potential temperature at the surface (top layer)
            # temp shape is typically (time, vertical_layer)
            sst_all = f['temp'][:, -1] 
            u_wind = f['Uwind'][:]
            v_wind = f['Vwind'][:]
            
            lat = f['lat_rho'][()]
            lon = f['lon_rho'][()]

        times = [base_time + timedelta(seconds=t) for t in ocean_time]
        # Calculate scalar wind speed from U and V components (Model units: m/s)
        wind_speed_ms = np.sqrt(u_wind**2 + v_wind**2)

        print(f"Coordinates: {lat:.4f} N, {lon:.4f} E (Sunset Point, San Juan Island)")
        print(f"{'Date':<12} | {'SST (C)':<8} | {'Wind (m/s)':<12} | {'Wind (km/hr)':<12}")
        print("-" * 65)

        for date_str in target_dates:
            try:
                target_dt = datetime.strptime(date_str, '%Y-%m-%d')
                noon_dt = target_dt.replace(hour=12)
                diffs = [abs((t - noon_dt).total_seconds()) for t in times]
                idx = np.argmin(diffs)
                
                if times[idx].date() == target_dt.date():
                    ws_ms = wind_speed_ms[idx]
                    ws_kmh = ws_ms * 3.6  # Conversion factor
                    print(f"{date_str:<12} | {sst_all[idx]:<8.2f} | {ws_ms:<12.2f} | {ws_kmh:<12.2f}")
                else:
                    print(f"{date_str:<15} | No model data available for this specific date.")
            except ValueError:
                print(f"{date_str:<15} | Invalid date format. Use YYYY-MM-DD.")
            
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Ensure the NetCDF file is in the script directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Add or remove dates from this list as needed
    dates_to_check = ['2026-01-23', '2026-01-02', '2025-11-14']
    get_weather_at_sunset_point(dates_to_check)


def add_environmental_annotation(ax, filename, x=1.25, y=0.05):
    """
    Fetches tide + weather metadata and adds a formatted text box
    directly to a matplotlib axis.
    """

    meta = get_environmental_metadata(filename)

    text = (
        f"Tide height: {meta['tide_height']:.2f} m\n"
        f"Tidal current: {meta['tidal_current']:.6f} m/s\n"
        f"Temperature: {meta['temperature']:.1f} °C\n"
        f"Wind speed: {meta['wind_speed']:.1f} km/h"
    )

    ax.text(
        x, y,
        text,
        transform=ax.transAxes,
        fontsize=16,
        va='bottom',
        ha='left',
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.85,
            edgecolor="gray"
        )
    )

def bis_time_filter(trace, fs, nperseg=512, apply_mask=True):
    """Reconstructs a time series after applying BIS/MBS in freq domain."""
    if not apply_mask:
        return trace
    noverlap = int(nperseg * 0.8)
    f, t, Zxx = stft(trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S_log = 10 * np.log10(np.abs(Zxx)**2 + 1e-12)
    S_denoised = S_log - np.nanmedian(S_log, axis=1, keepdims=True) # MBS
    S_denoised = S_denoised - np.nanmedian(S_denoised, axis=0, keepdims=True) # BIS
    mask = 10**(S_denoised/20)
    _, clean_trace = istft(Zxx * (mask / (np.max(mask)+1e-12)), fs=fs, nperseg=nperseg, noverlap=noverlap)
    return clean_trace[:len(trace)]

def get_advanced_spec(trace, fs, nperseg=512, bis=False):
    """Spectrogram with optional BIS."""
    f, t, Zxx = stft(trace, fs=fs, nperseg=nperseg, noverlap=int(nperseg*0.8))
    S_log = 10 * np.log10(np.abs(Zxx)**2 + 1e-12)
    S_denoised = S_log - np.nanmedian(S_log, axis=1, keepdims=True) # MBS
    if bis:
        S_denoised = S_denoised - np.nanmedian(S_denoised, axis=0, keepdims=True)
    return f, t, S_denoised

def get_h5_metadata(fpath):
    """Extract basic metadata from a DAS HDF5 file."""
    with h5py.File(fpath, 'r') as fp:
        fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']
        dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
        gl = fp['Acquisition'].attrs['GaugeLength']
        ns, nx = fp['Acquisition']['Raw[0]']['RawData'].shape
    return {'fs': fs, 'dx': dx, 'gl': gl, 'ns': ns, 'nx': nx}

def load_data(fdir, flist):
    """Load and concatenate data from a list of H5 files."""
    meta = get_h5_metadata(os.path.join(fdir, flist[0]))
    ns, nx = meta['ns'], meta['nx']
    data = np.zeros((ns * len(flist), nx))
    for i, fname in enumerate(flist):
        with h5py.File(os.path.join(fdir, fname), 'r') as fp:
            data[i*ns : (i+1)*ns, :] = fp['Acquisition']['Raw[0]']['RawData']
    return data, meta

def preprocess(data, fs, lowcut=200):
    """Apply detrending and high-pass filtering."""
    data = detrend(data, axis=0)
    if lowcut > 0:
        b, a = sp.butter(2, lowcut, btype='highpass', fs=fs)
        data = sp.filtfilt(b, a, data, axis=0)
    return data

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Apply a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.butter(order, [low, high], btype='band')
    y = sp.filtfilt(b, a, data, axis=0)
    return y

def compute_sigma2(data_block, fs, nperseg, noverlap):
    """Compute mean power spectrum (sigma^2) across time for a block of channels."""
    stack = np.mean(data_block, axis=1)
    f, _, Zxx = stft(stack, fs=fs, nperseg=nperseg, noverlap=noverlap)
    sigma2 = np.mean(np.abs(Zxx)**2, axis=1)
    return f, sigma2[:, np.newaxis]

def plot_snr_comparison(t_ref, f_ref, snr_ref, 
                        t_loop, f_loop, snr_loop,
                        t_med, f_med, snr_med,
                        t_hat, f_hat, snr_hat,
                        title_suffix="", 
                        t_lim=(0, 30), f_lim=(50, 2500), 
                        vmin=0, vmax=14,
                        rect_win=None):
    """Plot 4-panel SNR comparison."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 24), dpi=300, sharex=True)
    
    datasets = [
        (t_ref, f_ref, snr_ref, "Reference Channel"),
        (t_loop, f_loop, snr_loop, f"Loop Stacked {title_suffix}"),
        (t_med, f_med, snr_med, "Median Subtraction (Post-Stack)"),
        (t_hat, f_hat, snr_hat, "SNR Hat (Denoised)")
    ]

    for ax, (t, f, data, title) in zip(axes, datasets):
        im = ax.pcolormesh(t, f, data, shading="auto", cmap="Greys", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01).set_label("SNR [dB]")
        ax.set_title(title)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlim(t_lim)
        ax.set_ylim(f_lim)

        if rect_win:
            t_win, f_win = rect_win
            rect = patches.Rectangle((t_win[0], f_win[0]), t_win[1]-t_win[0], f_win[1]-f_win[0],
                                     linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.grid(which='both', linestyle='--', alpha=0.5, color='black')

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    return fig, axes

def plot_mbs_comparison(t_ref, f_ref, snr_ref,
                        t_stack, f_stack, snr_stack,
                        t_hat, f_hat, snr_hat,
                        title_suffix="",
                        t_lim=(0, 30), f_lim=(50, 2500),
                        vmin=0, vmax=18,
                        rect_win=None):
    """Plot 3-panel SNR comparison focusing on Median Baseline Subtraction (MBS)."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 18), dpi=300, sharex=True)

    datasets = [
        (t_ref, f_ref, snr_ref, f"1. Ref Channel with MBS {title_suffix}"),
        (t_stack, f_stack, snr_stack, f"2. MBS on each channel then Stacked {title_suffix}"),
        (t_hat, f_hat, snr_hat, f"3. MBS on each channel then Noise Env Sub {title_suffix}")
    ]

    for ax, (t, f, data, title) in zip(axes, datasets):
        im = ax.pcolormesh(t, f, data, shading="auto", cmap="Greys", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01).set_label("SNR [dB]")
        ax.set_title(title)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlim(t_lim)
        ax.set_ylim(f_lim)

        if rect_win:
            t_win, f_win = rect_win
            rect = patches.Rectangle((t_win[0], f_win[0]), t_win[1]-t_win[0], f_win[1]-f_win[0],
                                     linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.grid(which='both', linestyle='--', alpha=0.5, color='black')

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    return fig, axes

def moving_average(x, w):
    """
    Apply a simple moving average filter to 1D or 2D array along the last axis.
    """
    if x.ndim == 1:
        return np.convolve(x, np.ones(w), 'same') / w
    else:
        return np.apply_along_axis(lambda m: np.convolve(m, np.ones(w)/w, mode='same'), axis=-1, arr=x)

def compute_snr_hat(signal_trace, noise_trace, window_samples=100, normalize=True):
    """
    Implements the SNR hat method from Goestchel et al. (2025).
    Denotes SNR calculation where the signal envelope is normalized by a smoothed 
    noise envelope estimated from a secondary frequency band (e.g., 32-48 Hz).
    
    Parameters
    ----------
    signal_trace : np.ndarray
        Strain data filtered in the target band (e.g., 14-30 Hz). [channel x time]
    noise_trace : np.ndarray
        Strain data filtered in the noise estimation band (e.g., 32-48 Hz). [channel x time]
    window_samples : int
        Size of the moving average window for smoothing the noise envelope (default 100).
    normalize : bool
        If True, normalize signal and noise by their respective variances (original Goestchel et al. 2025).
        If False, uses raw power ratios, aligning with standard SNR.
        
    Returns
    -------
    snr_hat : np.ndarray
        SNR values in dB.
    """
    # Envelopes using Hilbert transform
    s_env = np.abs(hilbert(signal_trace, axis=-1))
    n_env = np.abs(hilbert(noise_trace, axis=-1))
    
    # Smooth noise envelope
    n_env_bar = moving_average(n_env, window_samples)
    
    # Standard deviations for normalization (per channel)
    eps = 1e-12
    if normalize:
        sigma_s = np.std(signal_trace, axis=-1, keepdims=True)
        sigma_n = np.std(noise_trace, axis=-1, keepdims=True)
        term_s = (s_env / (sigma_s + eps))**2
        term_n = (n_env_bar / (sigma_n + eps))**2
    else:
        term_s = s_env**2
        term_n = n_env_bar**2
    
    snr_hat = 10 * np.log10(term_s / (term_n + eps))
    
    return snr_hat

def compute_snr_hat_spectrogram(signal_trace, noise_trace, fs, nperseg, noverlap, window_samples=100, normalize=True):
    """
    Compute a spectrogram where the power is normalized by the smoothed envelope 
    of a secondary noise band (SNR hat logic applied to TF domain).
    
    Parameters
    ----------
    signal_trace : np.ndarray
        1D time series (e.g., stacked or single channel).
    noise_trace : np.ndarray
        1D time series of the noise estimation band.
    fs : float
        Sampling frequency.
    nperseg, noverlap : int
        STFT parameters.
    window_samples : int
        Moving average window for the noise envelope.
    normalize : bool
        If True, normalize signal and noise by their respective variances.
        
    Returns
    -------
    f, t, SNR_hat_tf : np.ndarray
    """
    # STFT of signal
    f, t, Zxx = stft(signal_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Envelope of noise in time domain
    n_env = np.abs(hilbert(noise_trace))
    n_env_bar = moving_average(n_env, window_samples)
    
    # Interpolate smoothed envelope to match STFT time bins
    t_orig = np.arange(len(noise_trace)) / fs
    interp_func = interp1d(t_orig, n_env_bar, bounds_error=False, fill_value="extrapolate")
    n_env_stft = interp_func(t)
    
    # Normalization constants
    eps = 1e-12
    if normalize:
        sigma_s = np.std(signal_trace)
        sigma_n = np.std(noise_trace)
        term_s = np.abs(Zxx)**2 / (sigma_s**2 + eps)
        term_n = n_env_stft**2 / (sigma_n**2 + eps)
    else:
        term_s = np.abs(Zxx)**2
        term_n = n_env_stft**2
    
    SNR_hat_tf = 10 * np.log10(term_s / (term_n + eps))
    
    return f, t, SNR_hat_tf


def compute_snr_hat_f_dep(signal_trace, noise_trace, fs, nperseg, noverlap, window_samples=100):
    """
    Compute a spectrogram where the power is normalized by the smoothed 
    frequency-dependent envelope of a noise trace (n_env(f,t)).
    
    Parameters
    ----------
    signal_trace : np.ndarray
        1D time series (e.g., stacked signal).
    noise_trace : np.ndarray
        1D time series representing noise.
    fs : float
        Sampling frequency.
    nperseg, noverlap : int
        STFT parameters.
    window_samples : int
        Moving average window for smoothing the noise envelope.
        
    Returns
    -------
    f, t, SNR_hat_tf : np.ndarray
    """
    # 1. STFT of signal
    f, t, Zxx_sig = stft(signal_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 2. STFT of noise
    f_n, t_n, Zxx_noise = stft(noise_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 3. Noise envelope: magnitude of noise STFT smoothed in time
    n_env = np.abs(Zxx_noise)
    
    # If noise trace has different duration, interpolate
    if len(t) != len(t_n):
        interp_func = interp1d(t_n, n_env, axis=1, bounds_error=False, fill_value="extrapolate")
        n_env = interp_func(t)
        
    # Smooth along time axis
    n_env_bar = moving_average(n_env, window_samples)
    
    # 4. Compute SNR
    eps = 1e-12
    SNR_hat_tf = 10 * np.log10(np.abs(Zxx_sig)**2 / (n_env_bar**2 + eps))
    
    return f, t, SNR_hat_tf


def median_baseline_normalization(signal_trace, fs, nperseg, noverlap):
    """
    Compute STFT, normalize by median power baseline, and reconstruct time-domain signal.
    Useful for whitening/denoising before SNR calculation.
    """
    # Compute STFT
    f, t, Zxx = stft(signal_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Power spectrum
    Sxx = np.abs(Zxx) ** 2
    Sxx_log = 10 * np.log10(np.maximum(Sxx, 1e-12))

    # Median baseline (per frequency)
    median_log = np.median(Sxx_log, axis=1, keepdims=True)

    Sxx_log_normalized = Sxx_log - median_log

    # Convert median back to linear scale
    median_linear = 10 ** (median_log / 10)

    # Normalize complex STFT
    # Divide by sqrt of power to normalize the amplitude
    Zxx_normalized = Zxx / np.sqrt(median_linear + 1e-12)

    # ISTFT reconstruction
    _, signal_rec = istft(Zxx_normalized, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return signal_rec[:len(signal_trace)], f, t, Zxx_normalized

def snr_hat_normalization(signal_trace, noise_trace, fs, nperseg, noverlap, window_samples=100):
    """
    Apply SNR hat normalization (time-domain noise envelope) and reconstruct.
    """
    f, t, Zxx = stft(signal_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    n_env = np.abs(hilbert(noise_trace))
    n_env_bar = moving_average(n_env, window_samples)
    
    t_orig = np.arange(len(noise_trace)) / fs
    interp_func = interp1d(t_orig, n_env_bar, bounds_error=False, fill_value="extrapolate")
    n_env_stft = interp_func(t)
    
    eps = 1e-12
    Zxx_normalized = Zxx / (n_env_stft + eps)
    
    _, signal_rec = istft(Zxx_normalized, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return signal_rec[:len(signal_trace)], f, t, Zxx_normalized

def snr_hat_f_dep_normalization(signal_trace, noise_trace, fs, nperseg, noverlap, window_samples=100):
    """
    Apply frequency-dependent SNR hat normalization and reconstruct.
    """
    f, t, Zxx_sig = stft(signal_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_n, t_n, Zxx_noise = stft(noise_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    n_env = np.abs(Zxx_noise)
    if len(t) != len(t_n):
        interp_func = interp1d(t_n, n_env, axis=1, bounds_error=False, fill_value="extrapolate")
        n_env = interp_func(t)
        
    n_env_bar = moving_average(n_env, window_samples)
    
    eps = 1e-12
    Zxx_normalized = Zxx_sig / (n_env_bar + eps)
    
    _, signal_rec = istft(Zxx_normalized, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return signal_rec[:len(signal_trace)], f, t, Zxx_normalized

def compute_snr_spectrogram(signal_stack, sigma2, fs, nperseg, noverlap):
    """
    Compute SNR spectrogram.
    """
    f, t, Zxx = stft(
        signal_stack,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap
    )

    eps = 1e-12
    # Ensure sigma2 is broadcastable with Zxx (freq, time)
    if sigma2.ndim == 1:
        sigma2 = sigma2[:, np.newaxis]
    
    SNR_tf = 10 * np.log10(np.abs(Zxx)**2 / (sigma2 + eps))

    return f, t, SNR_tf

def median_subtraction_spectrogram(signal_trace, fs, nperseg, noverlap):
    """
    Compute median-subtraction spectrogram for SNR calculation (TFSNR Method 3/4).
    Subtracts the median log-power across time for each frequency bin.
    """
    # Compute STFT
    f, t, Zxx = stft(signal_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Power
    Sxx = np.abs(Zxx)**2
    Sxx_log = 10 * np.log10(np.maximum(Sxx, 1e-12))

    # Median subtraction (per frequency bin)
    median = np.median(Sxx_log, axis=1, keepdims=True)
    Sxx_dB = Sxx_log - median

    return f, t, Sxx_dB

def plot_tx(data, t, x, vmin=None, vmax=None, cmap='seismic', ax=None, 
            compute_envelope=False, title=None, norm=None):
    """
    Plot DAS data in time-distance (t-x) domain efficiently using imshow.
    
    Parameters
    ----------
    data : np.ndarray
        DAS data array [time x distance].
    t : np.ndarray
        Time vector.
    x : np.ndarray
        Distance/channel vector.
    vmin, vmax : float, optional
        Colorbar limits. For 0 to be white with 'seismic', set vmin=-vmax.
    cmap : str, optional
        Colormap. Diverging maps like 'seismic' or 'RdBu_r' work well for 0-centered data.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into.
    compute_envelope : bool, optional
        If True, computes the Hilbert envelope along the time axis.
    title : str, optional
        Plot title.
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap.
    """
    if compute_envelope:
        # Use next_fast_len for faster FFT-based Hilbert transform
        from scipy.fft import next_fast_len
        n_fft = next_fast_len(data.shape[0])
        data = np.abs(hilbert(data, N=n_fft, axis=0)[:data.shape[0], :])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.get_figure()

    # extent = [left, right, bottom, top]
    extent = [x[0], x[-1], t[0], t[-1]]
    
    im = ax.imshow(data, extent=extent, aspect='auto', cmap=cmap, norm=norm,
                   vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower')
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (s)')
    if title:
        ax.set_title(title)
    
    return fig, ax, im

def zero_shift(x, n):
    """
    Shift 1D array by n samples.
    positive -> delay
    negative -> advance
    """
    result = np.zeros_like(x)
    if n > 0:
        result[n:] = x[:-n]
    elif n < 0:
        result[:n] = x[-n:]
    else:
        result[:] = x
    return result

def fk_filt(data, tint, fs, xint, dx, c_min, c_max):
    """
    Apply f-k filter to data.
    """
    data_fft = fft2(detrend(data))
    # Make freq and wavenum vectors
    nx = data_fft.shape[1]
    ns = data_fft.shape[0]
    f = fftshift(fftfreq(ns, d=tint/fs))
    k = fftshift(fftfreq(nx, d=xint*dx))
    ff, kk = np.meshgrid(f, k)

    # Soundwaves have f/k = c so f = k*c
    g = 1.0 * ((ff <= kk * c_min) & (ff <= -kk * c_min))
    g2 = 1.0 * ((ff <= kk * c_max) & (ff <= -kk * c_max))

    g = g + np.fliplr(g)
    g2 = g2 + np.fliplr(g2)
    g = g - g2
    g = gaussian_filter(g, 1)

    g = (g - np.min(g.flatten())) / (np.max(g.flatten()) - np.min(g.flatten()))
    g = g.astype('f')

    data_fft_g = fftshift(data_fft) * np.transpose(g)
    data_g = ifft2(ifftshift(data_fft_g))
    
    return f, k, g, fftshift(data_fft), data_g


WHALE_DELAYS = {
    "humpback": {
        (0, 2): -1573, (0, 3): 450, (0, 4): -15, (0, 5): 1037, (0, 6): 2571, (0, 7): 3107,
        (1, 2): -136, (1, 3): 722, (1, 4): 893, (1, 5): 1083, (1, 6): 2639, (1, 7): 3171,
        (2, 3): -971, (2, 4): 3079, (2, 5): 4448, (2, 6): 5978, (2, 7): -1079,
        (3, 4): -1364, (3, 5): 656, (3, 6): 1392, (3, 7): 2429,
        (4, 5): 1369, (4, 6): 3781, (4, 7): 2843,
        (5, 6): 1538, (5, 7): 2079,
        (6, 7): 518
    },
    # "humpback": {
    #     (0, 1): -136,
    #     (0, 2): -1573, (0, 3): 450, (0, 4): -15, (0, 5): 1037, (0, 6): 2571, (0, 7): 3107,
    #     (1, 2): -136, (1, 3): 722, (1, 4): 893, (1, 5): 1083, (1, 6): 2639, (1, 7): 3171,
    #     (2, 3): -971, (2, 4): 3079, (2, 5): 4448, (2, 6): 5978, (2, 7): -1079,
    #     (3, 4): -1364, (3, 5): 656, (3, 6): 1392, (3, 7): 2429,
    #     (4, 5): 1369, (4, 6): 3781, (4, 7): 2843,
    #     (5, 6): 1538, (5, 7): 2079,
    #     (6, 7): 518
    # },
    "humpback_9-39": {
        (0, 2): 220, (0, 3): 881, (0, 4): 1306, (0, 5): -29, (0, 6): 2586, (0, 7): 2821,
        (1, 2): 207, (1, 3): 857, (1, 4): 1169, (1, 5): 1207, (1, 6): 2535, (1, 7): 2781,
        (2, 3): 650, (2, 4): 772, (2, 5): 992, (2, 6): 2323, (2, 7): 3,
        (3, 4): 144, (3, 5): 5, (3, 6): 1515, (3, 7): 1748,
        (4, 5): 30, (4, 6): 1369, (4, 7): 1612,
        (5, 6): 1341, (5, 7): 1579,
        (6, 7): 238
    },
    "biggs": {
        (0, 2): 1, (0, 3): -2156, (0, 4): 293, (0, 5): 192, (0, 6): 0, (0, 7): 1,
        (1, 2): -4, (1, 3): -1269, (1, 4): -953, (1, 5): 0, (1, 6): 0, (1, 7): -299,
        (2, 3): -1266, (2, 4): -450, (2, 5): -780, (2, 6): 751, (2, 7): -1572,
        (3, 4): 468, (3, 5): 1969, (3, 6): 465, (3, 7): 548,
        (4, 5): -3, (4, 6): -1, (4, 7): 0,
        (5, 6): 0, (5, 7): 1,
        (6, 7): 1
    },
    "jpod jan": {
        (0, 2): 188, (0, 3): -100, (0, 4): -86, (0, 5): -2134, (0, 6): 2367, (0, 7): 2388,
        (1, 2): 177, (1, 3): 2417, (1, 4): 1445, (1, 5): 1975, (1, 6): 2361, (1, 7): 2394,
        (2, 3): 251, (2, 4): 846, (2, 5): 1793, (2, 6): 2189, (2, 7): 2210,
        (3, 4): -368, (3, 5): -72, (3, 6): 477, (3, 7): 76,
        (4, 5): 556, (4, 6): 440, (4, 7): 966,
        (5, 6): 398, (5, 7): 400,
        (6, 7): 45
    }
}


def beamform_loops(data_dict, loops, event, normalize=False, return_aligned=False):
    """
    Beamform DAS loops using hardcoded delay dictionary.

    Parameters:
    -----------
    data_dict : dict
        {loop_number: 1D numpy array}
    loops : list[int]
        loops to beamform (first is reference)
    event : str
        'humpback', 'biggs', 'jpod jan'
    normalize : bool
        divide by number of loops
    return_aligned : bool
        return individual aligned traces

    Returns:
    --------
    beamformed_signal : np.ndarray
    aligned (optional) : dict
    """

    event = event.lower()
    if event not in WHALE_DELAYS:
        raise ValueError(f"No delays found for event: {event}")

    delays = WHALE_DELAYS[event]
    ref_loop = loops[0]

    if ref_loop not in data_dict:
        raise KeyError(f"Reference loop {ref_loop} missing in data_dict")

    # initialize beamformed signal
    bf_signal = np.copy(data_dict[ref_loop]).astype(float)

    aligned_dict = {ref_loop: bf_signal.copy()} if return_aligned else None
    count = 1

    for loop_b in loops[1:]:
        if loop_b not in data_dict:
            print(f"Skipping loop {loop_b}: not in data_dict")
            continue

        # find delay
        if (ref_loop, loop_b) in delays:
            shift = delays[(ref_loop, loop_b)]
        elif (loop_b, ref_loop) in delays:
            shift = -delays[(loop_b, ref_loop)]
        else:
            print(f"Warning: no delay for ({ref_loop}, {loop_b})")
            continue

        sig = data_dict[loop_b]

        # apply shift
        shifted = np.roll(sig, -shift)

        # zero wrap (important for beamforming)
        if shift > 0:
            shifted[-shift:] = 0
        elif shift < 0:
            shifted[:-shift] = 0

        bf_signal += shifted
        count += 1

        if return_aligned:
            aligned_dict[loop_b] = shifted

        print(f"Aligned loop {loop_b} → {ref_loop}, shift = {shift} samples")

    if normalize:
        bf_signal /= count

    if return_aligned:
        return bf_signal, aligned_dict

    return bf_signal


def k_notch_filter(data, k_width=1):
    """
    Remove synchronous common-mode noise by zeroing out the k=0 strip in the f-k domain.
    Preserves propagating and aliased signals.
    """
    nt, nx = data.shape
    data_fk = fftshift(fft2(data))
    mid_k = nx // 2
    data_fk[:, mid_k - k_width : mid_k + k_width + 1] = 0
    return ifft2(ifftshift(data_fk)).real

def beamform_pws(data, fs, dx, velocity, nu=1.5):
    """
    Perform frequency-domain beamforming (phase-alignment) and Phase-Weighted Stacking (PWS).
    Corrected phase sign: +1j to advance delayed signals.
    """
    nt, nx = data.shape
    if np.isinf(velocity) or velocity > 1e6:
        linear = np.mean(data, axis=1)
        shifted_traces = data
    else:
        data_fft = fft(data, axis=0)
        freqs = fftfreq(nt, d=1/fs)
        x = np.arange(nx) * dx
        shifts = np.exp(1j * 2 * np.pi * freqs[:, None] * (1.0/velocity) * x[None, :])
        shifted_traces = ifft(data_fft * shifts, axis=0).real
        linear = np.mean(shifted_traces, axis=1)
    
    if nu > 0:
        analytic = hilbert(shifted_traces, axis=0)
        phases = np.exp(1j * np.angle(analytic))
        coh = np.abs(np.mean(phases, axis=1))
        return linear, linear * (coh ** nu), coh
    return linear, linear, np.ones(nt)

def compute_bis_spectrogram(trace, fs, nperseg=512, noverlap=None, bis=True, mbs=True):
    """
    Compute a spectrogram with Median Spectral Baseline (MBS) and 
    Broadband Impulse Suppression (BIS) normalization.
    """
    if noverlap is None:
        noverlap = int(nperseg * 0.8)
    f, t, Zxx = stft(trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S_log = 10 * np.log10(np.abs(Zxx)**2 + 1e-12)
    
    S_denoised = S_log
    if mbs:
        # Subtract median across time (removes steady noise)
        S_denoised = S_denoised - np.nanmedian(S_denoised, axis=1, keepdims=True)
    if bis:
        # Subtract median across frequency (removes vertical spikes)
        S_denoised = S_denoised - np.nanmedian(S_denoised, axis=0, keepdims=True)
        
    return f, t, S_denoised
