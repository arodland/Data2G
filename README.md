# Data2G

Data2G is an HF data modem for amateur radio operators with speeds and robustness rivaling VARA HF and Pactor. It supports connected (ARQ) and unconnected/broadcast (FEC) operation, with effective one-way speeds as high as 6400 bps (48,000 bytes per minute, 8000 words per minute) at 25+ dB SNR, and sensitivity down to around -10 dB, albeit at 0.5% of the speed. It's suitable for chat, BBSes, email delivery, or APRS. 2400Hz is required for full speed, but Data2G can be configured for a maximum bandwidth of 500Hz to fit within narrowband segments of the bandplan.

## Versatile

Data2G has a VARA-compatible TNC for connected mode (Winlink, BBS), and a KISS TNC for applications like HF APRS. But there is also an extended non-connected mode where apps can send frames to designated "broadcast groups", with full control over the mode they're using, and subscribe to those groups on KISS ports. This lets you build chat or other sorts of interesting apps on top of Data2G without having to stick to the in-order guaranteed-delivery ARQ model.

There is also an experimental option to detect connected-mode AX.25 over KISS and enable automatic rate shifting using metadata injected into the burst headers, so that you can go much faster than the default robust KISS mode, under good conditions.

It's even possible to do reliable one-way bulk data transfer, a la FLAMP.

## Monitorable

The spec is open, and Data2G comes with a built-in monitor mode that shows all decoded packets, with session reconstruction, and sender and receiver callsigns if you've heard the CONNECT or ID bursts.

## Modern Techniques

Data2G uses a number of advanced techniques, and all of them have been extensively tested and optimized through
simulated end-to-end QSOs to prove their value.

* **Orthogonal Frequency-Division Multiplexing**: OFDM with pilots for equalization and cyclic prefix for multipath resistance is hardly a new idea anymore, but it's incredibly reliable and widely used.
* **Crest-Factor Control**: The biggest problem with OFDM, especially for hams, is that combining multiple carriers increases "crest factor", or peak-to-average power ratio. Our equipment and our regulatory limits cap our peak envelope power (PEP), and a high PAPR means a low average power, which means less energy received, which means less bitrate (or less reliability). Data2G uses an advanced iterative clipper tuned to every single submode to give the best "PEP-fair" sensitivity. All of VARA HF's data modes have 9 dB PAPR, while Data2G's range from 0.14 dB to 8.38 dB, with 30% being above 4 dB. This translates directly into reliability.
* **Optimized Constellations**: Instead of square constellations, the 64-QAM and 256-QAM modes use constellations that are optimized for SNR through the clipper, giving them lower PAPR at their optimal decode points.
* **Active Constellation Extension**: 16-QAM doesn't benefit from the above tuning. Instead, 16-QAM modes work with the clipper to push distortion onto the 12 "outer" constellation points (which are free move away from the center without harming decode accuracy) and keep it off of the 4 "inner" points (which have neighbors on all sides and can't afford to move at all). This allows tighter clipping, and lower PAPR.
* **Long-block QC-LDPC codes**: Similar to the ones used in WiFi and 5G, these provide incredibly strong error recovery on the data frames with an affordable soft-decoding algorithm.
* **Polar codes**: Also found in 5G, these protect acknowledgments (and a few data submodes) with even higher redundancy.
* **Hybrid ARQ**: When noise and fading prevent a codeword from being decoded, the re-sent codeword carries different parity bits from what was sent the first time. The receiver can soft-combine the two copies of the data bits with the two different sets of parity bits, giving an even higher chance of recovery than with a simple resend.
* **Decision-Directed Channel Re-Estimation**: With enough CPU, every symbol can be an equalization pilot. Pulls a few dB out of thin air under fast QSB, improving usable transfer rates by 5 to 20 percent in some cases.
* **Transparent Compression**: Text takes up less airtime, automatically, with no cost to binary data.
* **Intelligent Rate Shifting**: The receiver feeds the current band conditions to the world's tiniest neural net model (61,280 parameters) which was trained on thousands of simulated sessions spanning over a quarter million transmissions in varying conditions. The model is used to predict which mode has the best balance of speed and probability of being decoded.
* **Submode Zoo**: Many modems have a carefully crafted set of modulations, arranged in a "ladder" from slowest and most robust to fastest and most fragile, so that their rate-shifters can "climb the ladder" when conditions are good and go back down it when conditions are poor. Data2G has 50 submodes spanning four bandwidths, seven modulations, and a gamut of parity levels. They don't fall into a "ladder", but each one is the best possible mode under _some_ set of conditions, whether that's high SNR, low SNR, fast fading, slow fading, or acknowledging packets as fast as possible. If it isn't somewhere on the Pareto frontier, it gets dropped.
