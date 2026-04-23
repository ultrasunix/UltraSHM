#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Created on Mon Apr 13 15:21:52 2026
@author: Xiaoyu Sun @ HGMRI
"""

"""
TiePie HS5 pitch-catch guided-wave measurement
----------------------------------------------
- AWG output: Gaussian-windowed chirp, 100 kHz -> 300 kHz
- Receiver: CH1
- Real-time refresh every 0.5 s
- Stop by pressing any key (Windows console)

Based on TiePie example patterns:
- GeneratorArbitrary.py
- OscilloscopeBlockNumpy.py
- OscilloscopeGeneratorTrigger.py
"""

import sys
sys.path.append('/home/xs/ultra/lib/python3.13/site-packages') # Add LibTiePie
import os
import time
import array
import spidev
import libtiepie
import numpy as np
import matplotlib.pyplot as plt
from array import array
from scipy.io import savemat
from scipy.signal import butter, filtfilt
from scipy.signal.windows import tukey
from gpiozero import LED # DigitalOutputDevice
from fn_create_gaussian_chirp import *  # Upload waveform Functions


# ----------------------------- User-adjustable parameters -----------------------------
PULSE_FREQ_START = 2e6   # Chirp Frequency Start
PULSE_FREQ_STOP = 8e6    # Chirp Frequency End
PULSE_DURATION = 80e-6   # 50 us (5-cycle at 5 MHz is 1 us)
AWG_OFFSET = 0.0         # V_offset
AWG_AMPLITUDE = 8.0     # V_peak, i.e. 2.0 means +/-2 V, which is 4 Vpp open-circuit
AWG_SAMPLE_FREQ = 100e6  # 100 MSa/s for arbitrary waveform synthesis
CH1_RANGE = 2.0          # +/-2 V range; increase if clipping occurs
SCP_SAMPLE_FREQ = 100e6  # 100 MSa/s Sampling frequency for oscilloscope acquisition; must be >= 2x F_STOP for Nyquist
SCP_SAMPLE_RATE = 14     # 14-bit vertical resolution; adjust if needed
ACQ_DURATION = 300e-6    # 1.0 ms capture window
PRETRIGGER_RATIO = 0.1   # 10% pre-trigger
REFRESH_INTERVAL = 0.5  # seconds
PH_VELOCITY = 5850       # Material Ph_velocity m/s
TX_MIN = 1               # Minimum Tx CHannel
TX_MAX = 32              # Maximum Tx CHannel
RX_MIN = 1               # Minimum Rx CHannel
RX_MAX = 32              # Maximum Rx CHannel
ACQ_NUM = 2              # Average Number
MUX_CHIP_NUM = 4         # current max is 4
MUX_CHANNEL_NUM = 16     # current max is 16
SAVE_DATA = True
HAVE_PRINTINFO = False
NUM_TIME_POINTS = int(np.ceil(ACQ_DURATION * SCP_SAMPLE_FREQ))  # Number of time points in acquisition
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILENAME = os.path.join(SCRIPT_DIR, "sdata-fmc-32els-5mhz-fbh-b3-aveg.mat")


# ---------- Connection Settings ----------
MXT1 = LED(5) # LE1 Chip Select T1
MXT2 = LED(6) # LE2 Chip Select T2
MXR1 = LED(27) # LE3 Chip Select R1
MXR2 = LED(22) # LE4 Chip Select R2
MXT1.off()
MXT2.off()
MXR1.off()
MXR2.off()
time.sleep(1)

spit = spidev.SpiDev()
spit.open(0,0)
spit.max_speed_hz = 500000  # 1 MHz SPI clock, safe for MAX14866
spit.mode = 0b00  # Mode 0

spir = spidev.SpiDev()
spir.open(1,0)
spir.max_speed_hz = 500000  # 1 MHz SPI clock, safe for MAX14866
spir.mode = 0b00  # Mode 0


# ---------- Functions ----------

def open_hs5_scope_and_generator():
    """
    Open an oscilloscope + generator from the same TiePie device,
    following the pattern of OscilloscopeGeneratorTrigger.py.
    """
    libtiepie.network.auto_detect_enabled = True
    libtiepie.device_list.update()

    scp = None
    gen = None

    for item in libtiepie.device_list:
        if item.can_open(libtiepie.DEVICETYPE_OSCILLOSCOPE) and item.can_open(libtiepie.DEVICETYPE_GENERATOR):
            candidate_scp = item.open_oscilloscope()
            if candidate_scp is None:
                continue

            if not (candidate_scp.measure_modes & libtiepie.MM_BLOCK):
                del candidate_scp
                continue

            candidate_gen = item.open_generator()
            if candidate_gen is None:
                del candidate_scp
                continue

            if not (candidate_gen.signal_types & libtiepie.ST_ARBITRARY):
                del candidate_gen
                del candidate_scp
                continue

            scp = candidate_scp
            gen = candidate_gen
            break

    if scp is None or gen is None:
        raise RuntimeError("No TiePie HS5-like device found with both oscilloscope block mode and arbitrary generator.")

    return scp, gen


def configure_generator(gen, waveform, fs_awg):
    """
    Configure AWG for arbitrary waveform burst output.
    """
    gen.signal_type = libtiepie.ST_ARBITRARY
    gen.frequency_mode = libtiepie.FM_SAMPLERATE
    gen.frequency = fs_awg
    gen.amplitude = AWG_AMPLITUDE
    gen.offset = AWG_OFFSET

    # Use burst mode when available, so each measurement emits one chirp packet.
    if gen.modes_native & libtiepie.GM_BURST_COUNT:
        gen.mode = libtiepie.GM_BURST_COUNT
        gen.burst_count = 1

    gen.output_enable = True

    data = array('f', waveform.astype(np.float32))
    gen.set_data(data)


def configure_scope(scp):
    """
    Configure oscilloscope CH1 in block mode.
    Trigger on generator new period.
    """
    scp.measure_mode = libtiepie.MM_BLOCK
    scp.sample_rate = SCP_SAMPLE_FREQ
    scp.resolution = SCP_SAMPLE_RATE
    scp.record_length = NUM_TIME_POINTS
    scp.pre_sample_ratio = PRETRIGGER_RATIO

    # Disable all channels first
    for ch in scp.channels:
        ch.enabled = False

    # Enable CH1 only
    ch1 = scp.channels[0]
    ch1.enabled = True
    ch1.range = CH1_RANGE
    ch1.coupling = libtiepie.CK_DCV

    # Triggering is block-mode only on HS5 manual
    scp.trigger.timeout = 0.2  # 200 ms timeout

    # Disable all trigger sources first
    for ch in scp.channels:
        if ch.has_trigger:
            ch.trigger.enabled = False

    for trig_in in scp.trigger_inputs:
        trig_in.enabled = False

    # Trigger from generator new period, like TiePie example
    trig_in = scp.trigger_inputs.get_by_id(libtiepie.TIID_GENERATOR_START) # or TIID_GENERATOR_START or TIID_GENERATOR_STOP or TIID_GENERATOR_NEW_PERIOD
    if trig_in is None:
        raise RuntimeError("Internal trigger source not available.")
    trig_in.enabled = True


def acquire_one_trace(scp, gen):
    """
    Start one block acquisition and one AWG burst, then return CH1 data.
    """
    scp.start()
    gen.start()

    t0 = time.time()
    while not scp.is_data_ready:
        if scp.is_data_overflow:
            raise RuntimeError("Oscilloscope data overflow.")
        if (time.time() - t0) > 1.0:
            raise TimeoutError("Timed out waiting for oscilloscope data.")
        time.sleep(0.001)

    data = scp.get_data_numpy()

    # Ensure generator is not left running
    try:
        gen.stop()
    except Exception:
        pass

    # Only CH1 is enabled, but get_data_numpy() may still return 2D array
    # Expected shape: (channels, samples)
    if data.ndim == 2:
        ch1 = data[0]
    else:
        ch1 = data

    return np.asarray(ch1, dtype=np.float64)


def apply_highpass_iir(x, fs, cutoff=20e3, order=4):
    nyq = 0.5 * fs
    wn = cutoff / nyq
    b, a = butter(order, wn, btype='highpass')
    y = filtfilt(b, a, x)
    return y, b, a


def save_data_mat(filename, t, data, signal, signal_amp, scope_sens, ph_velocity, tx, rx, buttera, butterb):

    exp_data = {
        'exp_data': {
            'in_chirped_signal': np.asarray(signal).reshape(-1, 1),
            'time': np.asarray(t).reshape(-1, 1),
            'time_data': np.asarray(data),   # keep 2D (N × cycles)
            'tx': np.asarray(tx).reshape(-1, 1),
            'rx': np.asarray(rx).reshape(-1, 1),
            'ph_velocity': np.array([[ph_velocity]]),
            'pf_butter_a': np.array([[buttera]]),
            'pf_butter_b': np.array([[butterb]]),
            'signal_amp': np.array([[signal_amp]]),   # scalar → 1x1
            'scope_sens': np.array([[scope_sens]]),
        }
    }

    savemat(filename, exp_data)
    time.sleep(1)
    return print(f"\033[92m---Data is saved---\033[0m")


def fmc_sequence(n_elements: int, include_self: bool = True, one_based: bool = False):
    """
    Generate FMC (Full Matrix Capture) TX/RX pairs.

    Returns:
      tx_seq, rx_seq: 1D int arrays of equal length
        Each entry k corresponds to one acquisition:
          transmit element = tx_seq[k], receive element = rx_seq[k]

    include_self: True -> include (tx==rx) pairs
    one_based: True -> indices are 1..N, else 0..N-1
    """
    idx = np.arange(n_elements, dtype=int)

    # All pairs: TX major order (tx fixed, rx sweeps)
    tx_seq = np.repeat(idx, n_elements)
    rx_seq = np.tile(idx, n_elements)

    if not include_self:
        mask = tx_seq != rx_seq
        tx_seq = tx_seq[mask]
        rx_seq = rx_seq[mask]

    if one_based:
        tx_seq = tx_seq + 1
        rx_seq = rx_seq + 1

    return tx_seq, rx_seq


def set_switches(HEX_CHT, HEX_CHR, SWMX):
    
    # Transmission
    if SWMX[0]:
        MXT1.off()
    elif SWMX[1]:
        MXT2.off()
        
    
    msb = (HEX_CHT >> 8) & 0xFF
    lsb = HEX_CHT & 0xFF
    time.sleep(0.001)
    spit.xfer([msb, lsb]) # Switch TMUX
    time.sleep(0.001)
    
    if SWMX[0]:
        MXT1.on()
    elif SWMX[1]:
        MXT2.on()
    
    time.sleep(0.001)
    
    # Reception
    if SWMX[2]:
        MXR1.off()
    elif SWMX[3]:
        MXR2.off()
    
    msb = (HEX_CHR >> 8) & 0xFF
    lsb = HEX_CHR & 0xFF
    time.sleep(0.001)
    spir.xfer([msb, lsb]) # Switch RMUX
    time.sleep(0.001)
    
    if SWMX[2]:
        MXR1.on()
    elif SWMX[3]:
        MXR2.on()
        
    time.sleep(0.001)

def close_switches(HEXN):
    
    # Transmission
    MXT1.off()
    MXT2.off()
    MXR1.off()
    MXR2.off()
    
    msb = (HEXN >> 8) & 0xFF
    lsb = HEXN & 0xFF
    time.sleep(0.001)
    spit.xfer([msb, lsb]) # Switch TMUX
    time.sleep(0.001)
    spir.xfer([msb, lsb]) # Switch RMUX
    time.sleep(0.001)
    
    MXT1.on()
    MXT2.on()
    MXR1.on()
    MXR2.on()
    
    time.sleep(0.001)


def main():
    if HAVE_PRINTINFO:
        print_library_info()

    print("Pitch-catch guided-wave measurement")
    print(f"Excitation      : Windowed Chirp from {PULSE_FREQ_START/1e3:.0f} kHz - {PULSE_FREQ_STOP/1e3:.0f} kHz")
    print(f"Pulse duration  : {PULSE_DURATION*1e6:.1f} us")
    print(f"Acquisition     : {ACQ_DURATION*1e3:.2f} ms at {SCP_SAMPLE_FREQ/1e6:.2f} MSa/s")
    print("Press any key in the console to stop.\n")

    # ---------- Preset Hex Channel Numbers ----------
    array_binary = np.flipud(np.eye(MUX_CHANNEL_NUM, dtype=int))
    array_hex = []
    for ii in range(0, MUX_CHANNEL_NUM, 1):
        array_hex.append(hex(int(''.join(map(str, array_binary[ii,:])),2)))
    array_hex.extend(array_hex)

    # %% ---------- Preset Tx-Rx Sequences --------------
    num_els = int(MUX_CHANNEL_NUM*(MUX_CHIP_NUM/2))#32
    tx, rx = np.indices((num_els, num_els))
    tx = tx.flatten() + 1 # with pulse-echo
    rx = rx.flatten() + 1 # with pulse-echo
    # tx = tx[~np.eye(tx.shape[0], dtype=bool)].reshape(tx.shape[0], -1).flatten() + 1 # without pulse-echo
    # rx = rx[~np.eye(rx.shape[0], dtype=bool)].reshape(rx.shape[0], -1).flatten() + 1 # without pulse-echo
    # tx, rx = fmc_sequence(num_els, True, True)
    IMXT1 = np.array([1] * (len(tx) // 2) + [0] * (len(tx) // 2), dtype=int)
    IMXT2 = np.array([0] * (len(tx) // 2) + [1] * (len(tx) // 2), dtype=int)
    IMXR1 = np.tile(np.array([1]*(num_els//2)+[0]*(num_els//2),dtype=int), (1,num_els))
    IMXR2 = np.tile(np.array([0]*(num_els//2)+[1]*(num_els//2),dtype=int), (1,num_els))
    IMXR1 = IMXR1.flatten()
    IMXR2 = IMXR2.flatten()
    ITOT = np.stack((IMXT1, IMXT2, IMXR1, IMXR2), axis=1)
    del IMXT1, IMXT2, IMXR1, IMXR2, array_binary

    # %% ---------- Generate TX-RX Mask ----------
    mask = (tx >= TX_MIN) & (tx <= TX_MAX) & (rx >= RX_MIN) & (rx <= RX_MAX)
    tx = tx[mask]
    rx = rx[mask]

    _, awg_wave = fn_create_gaussian_chirp(AWG_SAMPLE_FREQ, PULSE_DURATION, PULSE_FREQ_START, PULSE_FREQ_STOP)
    t = np.arange(NUM_TIME_POINTS, dtype=np.float64) / SCP_SAMPLE_FREQ
    scp, gen = open_hs5_scope_and_generator()

    try:
        configure_generator(gen, awg_wave, AWG_SAMPLE_FREQ)
        configure_scope(scp)
        close_switches(0) # close all channels by setting as 0x00
        time.sleep(1)

        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 9))
        line, = ax.plot(t * 1e6, np.zeros_like(t))
        ax.set_ylim(-CH1_RANGE, CH1_RANGE)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("HS5-CH1 received FMC bulk-wave signal")
        ax.grid(True)

        annotation = ax.text(
            0.02, 0.95, "",
            transform=ax.transAxes,
            verticalalignment="top"
        )

        all_traces = []

        for tr in range(0, len(tx), 1):
            ii = num_els*(tx[tr]-1)+rx[tr]
            
            if ((tr>0) and (ITOT[(ii-1), 2] != ITOT[(idx), 2])) or (tr>0) and (ITOT[(ii-1), 0] != ITOT[(idx), 0]):
                close_switches(0) # close all channels by setting as 0x00
                
            idx = ii-1
            
            set_switches(int(array_hex[tx[tr]-1],16), int(array_hex[rx[tr]-1],16), ITOT[(ii-1),:])
            time.sleep(0.01)
            
            # GET DATA
            time_data = np.zeros((len(t),),dtype=float)
            
            for an in range(0, ACQ_NUM, 1):
                tmp, butterb, buttera = apply_highpass_iir(acquire_one_trace(scp, gen), SCP_SAMPLE_FREQ, cutoff=50e3, order=3)
                time_data += tmp
                
                line.set_ydata(time_data)
                ax.relim()
                ax.autoscale_view()
                peak = np.max(np.abs(time_data))
                annotation.set_text(
                    f"Trace: {tr+1}\n"
                    f"Tx: {tx[tr]} and Rx: {rx[tr]}\n"
                    f"Average: {an+1}\n"
                    f"Peak |V|: {peak:.2f} V\n"
                    f"Refresh: {REFRESH_INTERVAL:.2f} s"
                )

                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(REFRESH_INTERVAL)
            
            all_traces.append(time_data.copy())
        
        if SAVE_DATA and len(all_traces) > 0:
            data_matrix = np.column_stack(all_traces)   # shape: N x M
            save_data_mat(SAVE_FILENAME, t, data_matrix, awg_wave, AWG_AMPLITUDE, CH1_RANGE, PH_VELOCITY, tx, rx, buttera, butterb)

    finally:
        try:
            gen.stop()
        except Exception:
            pass

        try:
            gen.output_enable = False
        except Exception:
            pass

        try:
            scp.stop()
        except Exception:
            pass
        del gen
        del scp

        # Clear Assigned Pins
        MXT1.close()
        MXT2.close()
        MXR1.close()
        MXR2.close()
        spit.close()
        spir.close()

        print("Measurement stopped safely.")


if __name__ == "__main__":
    main()