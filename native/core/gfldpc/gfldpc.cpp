#include "gfldpc/gfldpc.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "util/simd.hpp"

namespace data2g::gfldpc {

namespace {

constexpr unsigned POLY[] = {0, 0, 0, 0b1011, 0b10011, 0b100101};  // gfldpc.POLY by p
constexpr float FLOOR = 1e-20f;

// Unnormalized Walsh-Hadamard transform in place (Sylvester order: scipy.linalg.hadamard's).
// Compile-time Q (8, 16, 32) unrolls the butterfly stages into SIMD registers.
template <int Q>
DATA2G_SIMD_CLONES void wht_q(double* __restrict a) {
    for (int h = 1; h < Q; h *= 2)
        for (int i = 0; i < Q; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const double x = a[j], y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
}

template <int Q>
DATA2G_SIMD_CLONES void mul_norm_q(float* __restrict v, const float* __restrict w) {
    float mx = 0.0f;
    for (int a = 0; a < Q; ++a) {
        const float p = v[a] * w[a];
        v[a] = p;
        mx = p > mx ? p : mx;
    }
    const float k = mx > 0.0f ? 1.0f / mx : 1.0f;
    for (int a = 0; a < Q; ++a) {
        const float s = v[a] * k;
        v[a] = s > FLOOR ? s : FLOOR;
    }
}

template <int Q>
DATA2G_SIMD_CLONES void mul_norm_into_q(const float* __restrict src, const float* __restrict w, float* __restrict dst) {
    float mx = 0.0f;
    for (int a = 0; a < Q; ++a) {
        const float p = src[a] * w[a];
        dst[a] = p;
        mx = p > mx ? p : mx;
    }
    const float k = mx > 0.0f ? 1.0f / mx : 1.0f;
    for (int a = 0; a < Q; ++a) {
        const float s = dst[a] * k;
        dst[a] = s > FLOOR ? s : FLOOR;
    }
}

template <int Q>
DATA2G_SIMD_CLONES void v2c_msg_q(const float* __restrict fwd_j, const float* __restrict tmp, float* __restrict m) {
    float s = 0.0f;
    for (int a = 0; a < Q; ++a) {
        const float p = fwd_j[a] * tmp[a];
        m[a] = p;
        s += p;
    }
    const float k = 1.0f / s;
    for (int a = 0; a < Q; ++a) m[a] *= k;
}

template <int Q>
DATA2G_SIMD_CLONES void cvec_mul(const double* __restrict a, const double* __restrict b, double* __restrict out) {
    for (int i = 0; i < Q; ++i) out[i] = a[i] * b[i];
}

template <int Q>
DATA2G_SIMD_CLONES void cvec_mul_inplace(double* __restrict a, const double* __restrict b) {
    for (int i = 0; i < Q; ++i) a[i] *= b[i];
}

}  // namespace

GF::GF(int p_) : p(p_), q(1 << p_), mul(static_cast<std::size_t>(q * q), 0), inv(static_cast<std::size_t>(q), 0) {
    if (p < 3 || p > 5) throw std::invalid_argument("GF(2^p): p must be 3, 4 or 5");
    std::vector<int> exp(static_cast<std::size_t>(q - 1)), log(static_cast<std::size_t>(q), 0);
    unsigned x = 1;
    for (int i = 0; i < q - 1; ++i) {
        exp[i] = static_cast<int>(x);
        log[x] = i;
        x <<= 1;
        if (x & static_cast<unsigned>(q)) x ^= POLY[p];
    }
    for (int a = 1; a < q; ++a) {
        for (int b = 1; b < q; ++b) mul[a * q + b] = static_cast<std::uint8_t>(exp[(log[a] + log[b]) % (q - 1)]);
        inv[a] = static_cast<std::uint8_t>(exp[(q - 1 - log[a]) % (q - 1)]);
    }
}

Code::Code(int p_, int n_, int n0_, int dmax_, std::span<const std::int16_t> var_, std::span<const std::uint8_t> h_)
    : p(p_), q(1 << p_), n(n_), n0(n0_ ? n0_ : n_), m(static_cast<int>(var_.size()) / dmax_), dmax(dmax_),
      var(var_.begin(), var_.end()), h(h_.begin(), h_.end()), gf_(p_) {
    if (var_.size() != h_.size() || var_.size() % static_cast<std::size_t>(dmax_))
        throw std::invalid_argument("gfldpc::Code: var and h must be m * dmax");
    build();
}

Code::Code(const Code& c, int n_) : p(c.p), q(c.q), n(n_), n0(std::min(c.n0, n_)), m(0), dmax(0), gf_(c.p) {
    // the rows within n_, as gfldpc.cut: their slots up to the widest kept row's
    std::vector<int> keep;
    for (int r = 0; r < c.m; ++r) {
        int top = -1, w = 0;
        for (int d = 0; d < c.dmax; ++d) {
            const int v = c.var[r * c.dmax + d];
            if (v >= 0) top = std::max(top, v), ++w;
        }
        if (top < n_) {
            keep.push_back(r);
            dmax = std::max(dmax, w);
        }
    }
    m = static_cast<int>(keep.size());
    for (int r : keep)
        for (int d = 0; d < dmax; ++d) {
            var.push_back(c.var[r * c.dmax + d]);
            h.push_back(c.h[r * c.dmax + d]);
        }
    build();
}

Code Code::cut(int n_) const { return Code(*this, n_); }

void Code::build() {
    // core rows (all symbols < n0) and extension rows by their own symbol
    std::vector<int> core;
    ext_rows_.assign(static_cast<std::size_t>(n - n0), -1);
    for (int r = 0; r < m; ++r) {
        int top = -1;
        for (int d = 0; d < dmax; ++d) top = std::max(top, var[r * dmax + d]);
        if (top < n0) core.push_back(r);
        else if (ext_rows_[top - n0] < 0) ext_rows_[top - n0] = r;
        else throw std::invalid_argument("gfldpc::Code: two extension rows own one symbol");
    }
    for (int r : ext_rows_)
        if (r < 0) throw std::invalid_argument("gfldpc::Code: an extension symbol without its row");
    // the core's reduced row-echelon form, pivots from the last column (gfldpc.GFCode._echelon)
    const int mc = static_cast<int>(core.size());
    std::vector<std::uint8_t> H(static_cast<std::size_t>(mc * n0), 0);
    for (int i = 0; i < mc; ++i)
        for (int d = 0; d < dmax; ++d) {
            const int v = var[core[i] * dmax + d];
            if (v >= 0) H[i * n0 + v] ^= h[core[i] * dmax + d];
        }
    const auto* mul = gf_.mul.data();
    int r = 0;
    std::vector<std::uint8_t> is_piv(static_cast<std::size_t>(n0), 0);
    for (int col = n0 - 1; col >= 0 && r < mc; --col) {
        int pr = -1;
        for (int i = r; i < mc; ++i)
            if (H[i * n0 + col]) {
                pr = i;
                break;
            }
        if (pr < 0) continue;
        if (pr != r) std::swap_ranges(H.begin() + pr * n0, H.begin() + (pr + 1) * n0, H.begin() + r * n0);
        const int iv = gf_.inv[H[r * n0 + col]];
        for (int j = 0; j < n0; ++j) H[r * n0 + j] = mul[iv * q + H[r * n0 + j]];
        for (int i = 0; i < mc; ++i) {
            const int f = H[i * n0 + col];
            if (i == r || !f) continue;
            for (int j = 0; j < n0; ++j) H[i * n0 + j] ^= mul[f * q + H[r * n0 + j]];
        }
        parity_.push_back(col);
        is_piv[col] = 1;
        ++r;
    }
    for (int c = 0; c < n0; ++c)
        if (!is_piv[c]) info_.push_back(c);
    const std::size_t k = info_.size();
    a_.resize(static_cast<std::size_t>(r) * k);
    for (int i = 0; i < r; ++i)
        for (std::size_t j = 0; j < k; ++j) a_[i * k + j] = H[i * n0 + info_[j]];
}

Mat<std::uint8_t> Code::encode(const Mat<std::uint8_t>& u) const {
    const std::size_t k = info_.size();
    if (u.cols != k) throw std::invalid_argument("gfldpc::Code::encode: expected (B, k)");
    const auto* mul = gf_.mul.data();
    Mat<std::uint8_t> x(u.rows, static_cast<std::size_t>(n), 0);
    for (std::size_t b = 0; b < u.rows; ++b) {
        std::uint8_t* xb = x[b];
        for (std::size_t j = 0; j < k; ++j) xb[info_[j]] = u[b][j];
        for (std::size_t i = 0; i < parity_.size(); ++i) {
            unsigned acc = 0;
            for (std::size_t j = 0; j < k; ++j) acc ^= mul[a_[i * k + j] * q + u[b][j]];
            xb[parity_[i]] = static_cast<std::uint8_t>(acc);
        }
        for (int e = 0; e < n - n0; ++e) {  // each extension symbol from its row, in order
            const int row = ext_rows_[e], own = n0 + e;
            unsigned acc = 0, hs = 0;
            for (int d = 0; d < dmax; ++d) {
                const int v = var[row * dmax + d];
                if (v < 0) continue;
                if (v == own) hs = h[row * dmax + d];
                else acc ^= mul[h[row * dmax + d] * q + xb[v]];
            }
            xb[own] = mul[gf_.inv[hs] * q + acc];
        }
    }
    return x;
}

bool Code::syndrome_ok(const std::uint8_t* x) const {
    const auto* mul = gf_.mul.data();
    for (int r = 0; r < m; ++r) {
        unsigned acc = 0;
        for (int d = 0; d < dmax; ++d) {
            const int v = var[r * dmax + d];
            if (v >= 0) acc ^= mul[h[r * dmax + d] * q + x[v]];
        }
        if (acc) return false;
    }
    return true;
}

Decoder::Decoder(const Code& code) : code_(code) {
    std::vector<std::vector<int>> cols(static_cast<std::size_t>(code.n));
    for (int e = 0; e < code.m * code.dmax; ++e)
        if (code.var[e] >= 0) cols[code.var[e]].push_back(e);
    col_ptr_.push_back(0);
    for (const auto& c : cols) {
        col_edges_.insert(col_edges_.end(), c.begin(), c.end());
        col_ptr_.push_back(static_cast<int>(col_edges_.size()));
    }
}

namespace {

template <int Q>
Decoded decode_q(const Code& c, const std::vector<int>& col_ptr, const std::vector<int>& col_edges,
                 const Mat<float>& ll, int iters, bool posterior) {
    const int n = c.n, D = c.dmax;
    const auto* mul = c.gf().mul.data();
    const auto* inv = c.gf().inv.data();
    Decoded out{Mat<std::uint8_t>(ll.rows, static_cast<std::size_t>(n)), std::vector<std::uint8_t>(ll.rows, 0),
                posterior ? Mat<float>(ll.rows, static_cast<std::size_t>(n * Q)) : Mat<float>()};
    const std::size_t E = static_cast<std::size_t>(c.m * D);
    int dv = 0;
    for (int v = 0; v < n; ++v) dv = std::max(dv, col_ptr[v + 1] - col_ptr[v]);
    std::vector<float> ch(static_cast<std::size_t>(n * Q)), c2v(E * Q), v2c(E * Q),
        fwd(static_cast<std::size_t>((std::max(D, dv) + 1) * Q));
    std::vector<float> last_post;
    if (posterior) last_post.resize(static_cast<std::size_t>(n * Q));
    std::vector<double> W(static_cast<std::size_t>(D * Q)), cf(static_cast<std::size_t>(D * Q));
    std::vector<int> act(static_cast<std::size_t>(D));
    for (std::size_t b = 0; b < ll.rows; ++b) {
        std::uint8_t* hard = out.hard[b];
        for (int v = 0; v < n; ++v) {  // exp(ll - its max)
            const float* l = ll[b] + v * Q;
            const float mx = *std::max_element(l, l + Q);
            for (int a = 0; a < Q; ++a) ch[v * Q + a] = std::max(std::exp(l[a] - mx), FLOOR);
        }
        std::fill(c2v.begin(), c2v.end(), 1.0f);
        bool ok = false;
        for (int it = 0;; ++it) {
            // symbols: the posterior's decision, and each edge's message (the others' product, sum 1)
            for (int v = 0; v < n; ++v) {
                const int* es = &col_edges[col_ptr[v]];
                const int d = col_ptr[v + 1] - col_ptr[v];
                std::copy_n(&ch[v * Q], Q, &fwd[0]);  // fwd[j]: ch times the first j edges'
                for (int j = 0; j < d; ++j)
                    mul_norm_into_q<Q>(&fwd[j * Q], &c2v[es[j] * Q], &fwd[(j + 1) * Q]);
                const float* post = &fwd[d * Q];
                hard[v] = static_cast<std::uint8_t>(std::max_element(post, post + Q) - post);
                if (posterior) std::copy_n(post, Q, &last_post[v * Q]);
                float tmp[Q];
                for (int a = 0; a < Q; ++a) tmp[a] = 1.0f;  // the product of the edges past j
                for (int j = d - 1; j >= 0; --j) {
                    v2c_msg_q<Q>(&fwd[j * Q], tmp, &v2c[es[j] * Q]);
                    if (j) mul_norm_q<Q>(tmp, &c2v[es[j] * Q]);
                }
            }
            ok = c.syndrome_ok(hard);
            if (ok || it == iters) break;
            // checks
            for (int r = 0; r < c.m; ++r) {
                int deg = 0;
                for (int d = 0; d < D; ++d) {
                    const int e = r * D + d;
                    if (c.var[e] < 0) continue;
                    act[deg] = e;
                    double* w = &W[deg * Q];
                    const std::uint8_t* mhi = mul + inv[c.h[e]] * Q;
                    const float* m = &v2c[e * Q];
                    for (int y = 0; y < Q; ++y) w[y] = m[mhi[y]];  // P_y[y] = P_x[h^-1 y]
                    wht_q<Q>(w);
                    ++deg;
                }
                if (deg == 0) continue;
                // products of the others' transforms: forward (0..deg-2), then backward
                std::copy_n(&W[0], Q, &cf[0]);
                for (int d = 1; d < deg - 1; ++d) cvec_mul<Q>(&cf[(d - 1) * Q], &W[d * Q], &cf[d * Q]);
                double ct[Q];
                for (int a = 0; a < Q; ++a) ct[a] = 1.0;  // the backward product past d
                for (int d = deg - 1; d >= 0; --d) {
                    const int e = act[d];
                    double R[Q];
                    if (d) cvec_mul<Q>(&cf[(d - 1) * Q], ct, R);
                    else std::copy_n(ct, Q, R);
                    wht_q<Q>(R);
                    const std::uint8_t* mhv = mul + c.h[e] * Q;
                    double rp[Q];
                    double mx = 0.0;
                    for (int a = 0; a < Q; ++a) {
                        const double v = R[mhv[a]];
                        rp[a] = v;
                        mx = v > mx ? v : mx;
                    }
                    const double k = mx > 0.0 ? 1.0 / mx : 1.0;
                    float* m = &c2v[e * Q];
                    for (int a = 0; a < Q; ++a) {
                        const double v = rp[a] > 0.0 ? rp[a] * k : 0.0;
                        m[a] = std::max(static_cast<float>(v), FLOOR);
                    }
                    if (d > 0) cvec_mul_inplace<Q>(ct, &W[d * Q]);
                }
            }
        }
        // Compute log posteriors once on exit rather than on every iteration.
        if (posterior) {
            for (int v = 0; v < n; ++v) {
                const float* post = &last_post[v * Q];
                const float mx = post[hard[v]];
                float* op = out.post[b] + v * Q;
                for (int a = 0; a < Q; ++a) op[a] = std::log(post[a] / mx);
            }
        }
        out.ok[b] = ok;
    }
    return out;
}

}  // namespace

// The same belief propagation as gfldpc.Decoder (flooding, the check in the
// Walsh-Hadamard domain), on probabilities in place of log-probabilities: no
// exp or log in the loop, decisions alike.
Decoded Decoder::decode(const Mat<float>& ll, int iters, bool posterior) const {
    const Code& c = code_;
    if (ll.cols != static_cast<std::size_t>(c.n * c.q)) throw std::invalid_argument("gfldpc::Decoder: expected (B, n * q)");
    if (c.q == 8) return decode_q<8>(c, col_ptr_, col_edges_, ll, iters, posterior);
    if (c.q == 16) return decode_q<16>(c, col_ptr_, col_edges_, ll, iters, posterior);
    return decode_q<32>(c, col_ptr_, col_edges_, ll, iters, posterior);
}

Mat<std::uint8_t> pack(const Mat<std::uint8_t>& bits, int p, int k) {
    Mat<std::uint8_t> out(bits.rows, static_cast<std::size_t>(k), 0);
    for (std::size_t b = 0; b < bits.rows; ++b)
        for (std::size_t i = 0; i < bits.cols && i < static_cast<std::size_t>(k * p); ++i)
            out[b][i / p] |= static_cast<std::uint8_t>((bits[b][i] & 1) << (p - 1 - static_cast<int>(i % p)));
    return out;
}

Mat<std::uint8_t> unpack(const Mat<std::uint8_t>& x, int p) {
    Mat<std::uint8_t> out(x.rows, x.cols * p);
    for (std::size_t b = 0; b < x.rows; ++b)
        for (std::size_t i = 0; i < x.cols * p; ++i) out[b][i] = (x[b][i / p] >> (p - 1 - static_cast<int>(i % p))) & 1;
    return out;
}

}  // namespace data2g::gfldpc
