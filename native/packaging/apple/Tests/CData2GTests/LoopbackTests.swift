import CData2G
import XCTest

/// A minimal application: it queues what the modem hands it and plays a tick at a time.
final class LoopbackHost {
    var lines: [String] = []
    var data = Data()
    var queue: [Float] = []
    /// Samples that really went out. Flushed ones are not counted.
    var played: UInt64 = 0
    var ptt = false
    var keys = 0
    var flushes = 0
    var txModes: [String] = []
    var rxSnr: [Float] = []
    var modem: OpaquePointer?

    func callbacks() -> D2gCallbacks {
        D2gCallbacks(
            tx_audio: { ctx, samples, n in
                let host = Unmanaged<LoopbackHost>.fromOpaque(ctx!).takeUnretainedValue()
                host.queue.append(contentsOf: UnsafeBufferPointer(start: samples, count: n))
            },
            tx_flush: { ctx in
                let host = Unmanaged<LoopbackHost>.fromOpaque(ctx!).takeUnretainedValue()
                host.queue.removeAll(keepingCapacity: true)
                host.flushes += 1
            },
            ptt: { ctx, on in
                let host = Unmanaged<LoopbackHost>.fromOpaque(ctx!).takeUnretainedValue()
                if on { host.keys += 1 }
                host.ptt = on
            },
            host_line: { ctx, line in
                let host = Unmanaged<LoopbackHost>.fromOpaque(ctx!).takeUnretainedValue()
                if let line { host.lines.append(String(cString: line)) }
            },
            host_data: { ctx, data, n in
                let host = Unmanaged<LoopbackHost>.fromOpaque(ctx!).takeUnretainedValue()
                if let data, n > 0 { host.data.append(data, count: n) }
            },
            log: nil,
            tx_report: { ctx, report in
                let host = Unmanaged<LoopbackHost>.fromOpaque(ctx!).takeUnretainedValue()
                if let r = report?.pointee, let mode = r.mode { host.txModes.append(String(cString: mode)) }
            },
            rx_report: { ctx, report in
                let host = Unmanaged<LoopbackHost>.fromOpaque(ctx!).takeUnretainedValue()
                if let r = report?.pointee, !r.lost { host.rxSnr.append(r.snr_db) }
            }
        )
    }
}

final class LoopbackTests: XCTestCase {
    static let tick = 960  // 20 ms at 48 kHz

    func testVersionAndDefaults() {
        let ver = String(cString: d2g_version())
        XCTAssertTrue(ver.hasPrefix("Data2G "), ver)
        var cfg = D2gConfig()
        d2g_config_default(&cfg)
        XCTAssertEqual(cfg.sample_rate, UInt32(D2G_RATE_DEFAULT))
        XCTAssertEqual(cfg.sample_rate % UInt32(D2G_RATE_BASE), 0)
        XCTAssertGreaterThan(cfg.ptt_on_delay_s, 0)
        XCTAssertGreaterThan(d2g_max_net_bps(UInt8(D2G_BW_2300)), d2g_max_net_bps(UInt8(D2G_BW_500)))
        XCTAssertGreaterThan(d2g_max_net_bps(UInt8(D2G_BW_500)), 0)
    }

    func testRefusals() {
        var cfg = D2gConfig()
        d2g_config_default(&cfg)
        var none = D2gCallbacks()
        XCTAssertNil(d2g_new(&cfg, &none, nil), "required callbacks missing")
        let host = LoopbackHost()
        var cbs = host.callbacks()
        cfg.sample_rate = 44100
        XCTAssertNil(d2g_new(&cfg, &cbs, Unmanaged.passUnretained(host).toOpaque()), "44100 Hz")
    }

    func testTwoModemSession() {
        let aHost = LoopbackHost()
        let bHost = LoopbackHost()
        var cfg = D2gConfig()
        d2g_config_default(&cfg)
        cfg.decode_threads = 1

        var aCbs = aHost.callbacks()
        var bCbs = bHost.callbacks()
        cfg.seed = 1
        let a = d2g_new(&cfg, &aCbs, Unmanaged.passUnretained(aHost).toOpaque())
        cfg.seed = 2
        let b = d2g_new(&cfg, &bCbs, Unmanaged.passUnretained(bHost).toOpaque())
        XCTAssertNotNil(a)
        XCTAssertNotNil(b)
        aHost.modem = a
        bHost.modem = b
        var freed = false
        defer {
            if !freed {
                d2g_free(a)
                d2g_free(b)
            }
        }

        d2g_command(a, "MYCALL W1AW")
        d2g_command(b, "MYCALL K2XYZ")
        d2g_command(b, "LISTEN ON")
        XCTAssertTrue(bHost.lines.contains("REGISTERED K2XYZ"), "\(bHost.lines)")
        d2g_command(a, "CONNECT W1AW K2XYZ")

        let payload = Data((0..<800).map { UInt8(($0 * 7) & 0xff) })
        var sent = false
        var disconnecting = false
        // the engine's bursts peak at 1.0: about unit RMS at this gain, over noise 12 dB down in 2.5 kHz
        // (tests/test_embed.cpp's channel)
        let gain: Float = 2.2
        let noise = Float((24000.0 / 2500.0 / pow(10.0, 1.2)).squareRoot())
        var rng = SystemRandomNumberGenerator()
        func channel(_ air: [Float]) -> [Float] {
            air.map { s in
                // two uniforms to a Gaussian (Box-Muller)
                let u = Float.random(in: Float.leastNonzeroMagnitude..<1, using: &rng)
                let v = Float.random(in: 0..<1, using: &rng)
                return gain * s + noise * (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
            }
        }
        var airA = [Float](repeating: 0, count: Self.tick)
        var airB = [Float](repeating: 0, count: Self.tick)

        for _ in 0..<(120 * 50) {
            // a keyed radio delivers no receive audio: time only
            if aHost.ptt {
                d2g_advance(a, UInt64(Self.tick))
            } else {
                d2g_rx_audio(a, channel(airB), Self.tick)
            }
            if bHost.ptt {
                d2g_advance(b, UInt64(Self.tick))
            } else {
                d2g_rx_audio(b, channel(airA), Self.tick)
            }
            airA = take(from: aHost, modem: a!)
            airB = take(from: bHost, modem: b!)

            if !sent && bHost.lines.contains(where: { $0.hasPrefix("CONNECTED") }) {
                payload.withUnsafeBytes { raw in
                    d2g_send(b, raw.bindMemory(to: UInt8.self).baseAddress, payload.count)
                }
                sent = true
            }
            if !disconnecting && aHost.data.count >= payload.count {
                XCTAssertEqual(aHost.data.prefix(payload.count), payload)
                d2g_command(a, "DISCONNECT")
                disconnecting = true
            }
            if disconnecting && aHost.lines.contains("DISCONNECTED") && bHost.lines.contains("DISCONNECTED")
                && !aHost.ptt && !bHost.ptt
            {
                break
            }
        }
        XCTAssertTrue(sent, "never connected; A lines=\(aHost.lines) B lines=\(bHost.lines)")
        XCTAssertTrue(disconnecting, "no data; A lines=\(aHost.lines) B lines=\(bHost.lines)")
        XCTAssertTrue(aHost.lines.contains("DISCONNECTED") && bHost.lines.contains("DISCONNECTED"))
        XCTAssertFalse(aHost.ptt || bHost.ptt, "a radio left keyed")
        XCTAssertGreaterThan(aHost.keys, 0)
        XCTAssertEqual(aHost.lines.filter { $0 == "PTT ON" }.count, aHost.keys)
        XCTAssertEqual(aHost.flushes + bHost.flushes, 0)
        XCTAssertFalse(bHost.txModes.isEmpty)
        XCTAssertTrue(aHost.rxSnr.contains { $0.isFinite }, "\(aHost.rxSnr)")

        freed = true
        d2g_free(a)
        d2g_free(b)
        XCTAssertEqual(aHost.flushes, 1, "free: tx_flush")
    }

    /// One tick out of the queue to the air, and the running total to the modem.
    private func take(from host: LoopbackHost, modem: OpaquePointer) -> [Float] {
        var out = [Float](repeating: 0, count: Self.tick)
        if host.ptt {
            let n = min(Self.tick, host.queue.count)
            if n > 0 {
                out.replaceSubrange(0..<n, with: host.queue.prefix(n))
                host.queue.removeFirst(n)
                host.played += UInt64(n)
            }
        }
        d2g_tx_played(modem, host.played)
        return out
    }
}
