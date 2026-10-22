import numpy as np
from scipy.signal import hilbert
from scipy.signal import hilbert, stft, istft


def moving_average(x, w):
    """
    Apply a simple moving average filter to 1D or 2D array along the last axis.
    """
    if x.ndim == 1:
        return np.convolve(x, np.ones(w), 'same') / w
    else:
        return np.apply_along_axis(lambda m: np.convolve(m, np.ones(w) / w, mode='same'), axis=-1, arr=x)


def compute_snr_hat(signal_trace, noise_trace, window_samples=100, normalize=True, return_trace=False, gain_power=1.0):
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
        If True, scale each envelope by its trace's standard deviation (Goestchel et al. 2025).
        If False, use raw envelope amplitudes.
    return_trace : bool
        If True, returns both the SNR values and the modified signal trace after envelope subtraction.
    gain_power : float
        Exponent on the subtraction gain (affects trace_modified only). 1.0 = linear
        subtraction; >1.0 suppresses noise more aggressively.

    Returns
    -------
    snr_hat : np.ndarray
        SNR values in dB.
    trace_modified : np.ndarray (optional)
        The signal trace after subtracting the smoothed noise envelope.
    """
    s_env = np.abs(hilbert(signal_trace, axis=-1))
    n_env = np.abs(hilbert(noise_trace, axis=-1))

    n_env_bar = moving_average(n_env, window_samples)

    eps = 1e-12
    if normalize:
        sigma_s = np.std(signal_trace, axis=-1, keepdims=True)
        sigma_n = np.std(noise_trace, axis=-1, keepdims=True)
        s_env_scaled = s_env / (sigma_s + eps)
        n_env_bar_scaled = n_env_bar / (sigma_n + eps)
    else:
        s_env_scaled = s_env
        n_env_bar_scaled = n_env_bar

    snr_hat = 10 * np.log10(s_env_scaled ** 2 / (n_env_bar_scaled ** 2 + eps))

    if return_trace:
        s_env_sub = np.maximum(0, s_env_scaled - n_env_bar_scaled)
        gain = (s_env_sub / (s_env_scaled + eps)) ** gain_power
        trace_modified = signal_trace * gain
        return snr_hat, trace_modified

    return snr_hat


def enhance_tonal_signals(aligned_data, fs, nperseg=512, pws_nu=2.0, apply_bis=True):
    """
    Consolidated workflow for enhancing weak tonal signals in DAS data.
    Implements Phase-Weighted Stacking (PWS) and Broadband Impulse Suppression (BIS).

    Parameters:
    -----------
    aligned_data : np.ndarray
        Array of shape (N_channels, N_samples) containing pre-aligned DAS traces.

    fs : float
        Sampling frequency in Hz.

    nperseg : int
        Window size for STFT and BIS filtering.

    pws_nu : float
        Power for Phase-Weighted Stacking.
        nu=0 is linear, nu=2 is typical for PWS.

    apply_bis : bool
        If True, applies Broadband Impulse Suppression to the resulting stacks.

    Returns:
    --------
    results : dict

        {
            'linear_stack': 1D time series (Linear average),
            'pws_stack': 1D time series (Phase-weighted stack),
            'linear_spec': 2D array (MBS-normalized spectrogram of linear stack),
            'pws_spec': 2D array (MBS-normalized spectrogram of PWS stack),
            'pws_bis_spec': 2D array (BIS-enhanced spectrogram of PWS stack),
            'f': freq axis,
            't': time axis
        }
    """
    lin_stack = np.mean(aligned_data, axis=0)

    analytic = hilbert(aligned_data, axis=1)

    phase_coherence = np.abs(
        np.mean(
            analytic / (np.abs(analytic) + 1e-12),
            axis=0
        )
    )

    pws_stack = lin_stack * phase_coherence ** pws_nu

    def clean_spec(trace):
        noverlap = int(nperseg * 0.8)

        f, t, Zxx = stft(
            trace,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap
        )

        S_log = 10 * np.log10(
            np.abs(Zxx) ** 2 + 1e-12
        )

        S_mbs = S_log - np.nanmedian(
            S_log,
            axis=1,
            keepdims=True
        )

        S_bis = S_mbs - np.nanmedian(
            S_mbs,
            axis=0,
            keepdims=True
        )

        return f, t, S_mbs, S_bis

    f, t, lin_mbs, _ = clean_spec(lin_stack)

    _, _, pws_mbs, pws_bis = clean_spec(pws_stack)

    return {
        'linear_stack': lin_stack,
        'pws_stack': pws_stack,
        'linear_spec': lin_mbs,
        'pws_spec': pws_mbs,
        'pws_bis_spec': pws_bis,
        'f': f,
        't': t
    }
