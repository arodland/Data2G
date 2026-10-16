"""The OFDM waveform and the DSP around it: a map of the area.

TX (modem.modulate_bits): constellation.modulate -> ofdm.Band.modulate_symbols
(preamble, header, frames) -> dsp.tx_condition (envelope clip and filter).

RX (modem.receive): dsp.to_baseband -> sync.acquire, or sync.StreamDetector
live (tnc) -> ofdm.Band.demod_window per symbol -> equalizer.estimate (pilots;
equalizer.refine for decision-directed passes) -> constellation.llr.

Here:
- dsp: to_baseband, freq_correct, tx_condition, papr_db.
- ofdm: band(name) -> Band: carriers, DFT matrices, pilot, preamble.
- sync: the preamble's matched filter (cfo_grid, unit_template,
  repeat_corrs, noise_level), detection and refinement (acquire), and its
  streaming form (StreamDetector).

Beside it, at the top level of data2g: equalizer (channel estimation),
constellation (point sets, mapping, LLRs), cpm (the constant-envelope
M-FSK family: its own sync, header and soft bits; TX through
dsp.tx_condition) and interference (a simulator for studies, not on the
live path).

Native counterparts: native/core/waveform/{dsp,ofdm,sync}, and
native/core/{equalizer,constellation,cpm}. native/core/dsp/ is not this
dsp: it emulates numpy/scipy (quantile, firwin, hilbert, FFTs) for every
native module. Under `pytest --native`, tests/conftest.py substitutes these
functions by name, private ones included (sync._raw_stat, _refine,
_crossings; equalizer._doppler_corr; cpm._energies, _shares, _peak_ratio):
renaming one means renaming its substitution and binding too.
"""
