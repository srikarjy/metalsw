#include <metal_stdlib>
using namespace metal;

// Inter-sequence Smith-Waterman (Gotoh affine gap), one thread per database
// sequence. Stage 2 (v0.3): both kernels below read substitution scores from
// a precomputed query profile (profile[residueIdx * queryLen + j] ==
// blosum62[residueIdx][query[j]]) instead of a residue-index + 2D-matrix
// lookup — one indexed read per DP cell instead of two.
//
// smith_waterman_score_int8 is the fast path: H/E/F run in int8_t, 4x the
// per-thread footprint of the int16 version, but protein alignment scores
// overflow int8 range (+/-127) routinely — any thread that would saturate
// sets its overflow flag and its result must be discarded and recomputed by
// smith_waterman_score (the int16 kernel, unchanged from Stage 1) on the
// host side. Threadgroup tuning: threadgroup size is chosen host-side from
// threadExecutionWidth() rather than just maxTotalThreadsPerThreadgroup().
//
// Stage 3 (v0.6): Wavefront (anti-diagonal) parallelism — one threadgroup
// per database sequence, 32 threads cooperate along anti-diagonals.
// E recurrence computed sequentially left-to-right using threadgroup memory.
// F recurrence uses threadgroup memory for up neighbor from k-1.
// Threadgroup size = 32 (one simdgroup). Requires queryLen ≤ 32.

#define MAX_QUERY_LEN 512
#define ALPHABET_SIZE 24
#define WAVEFRONT_THREADS 32

struct Params {
    uint queryLen;
    uint dbCount;
    int gapOpen;
    int gapExtend;
};

inline short profileLookup(constant short *profile, constant char *residueIndex,
                            uint queryLen, char dbResidue, uint j) {
    short idx = residueIndex[(uchar)dbResidue];
    return profile[(uint)idx * queryLen + j];
}

inline short max_s(short a, short b) {
    return a > b ? a : b;
}

kernel void smith_waterman_score(
    constant Params &params [[buffer(0)]],
    constant short *profile [[buffer(1)]],
    constant char *residueIndex [[buffer(2)]],
    device const char *dbSeqs [[buffer(3)]],
    device const uint *dbOffsets [[buffer(4)]],
    device const uint *dbLengths [[buffer(5)]],
    device int *outScores [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.dbCount) return;

    const uint dbLen = dbLengths[tid];
    const uint dbOffset = dbOffsets[tid];
    const uint queryLen = min(params.queryLen, (uint)MAX_QUERY_LEN);

    thread short prevH[MAX_QUERY_LEN + 1];
    thread short prevF[MAX_QUERY_LEN + 1];
    thread short curH[MAX_QUERY_LEN + 1];
    thread short curE[MAX_QUERY_LEN + 1];
    thread short curF[MAX_QUERY_LEN + 1];

    for (uint j = 0; j <= queryLen; ++j) {
        prevH[j] = 0;
        prevF[j] = 0;
    }

    short best = 0;
    for (uint i = 1; i <= dbLen; ++i) {
        curH[0] = 0;
        curE[0] = 0;
        curF[0] = 0;
        short diag = prevH[0];
        const char dbResidue = dbSeqs[dbOffset + i - 1];

        for (uint j = 1; j <= queryLen; ++j) {
            const short matchScore = diag + profileLookup(profile, residueIndex, queryLen, dbResidue, j - 1);
            curE[j] = max_s((short)(curH[j - 1] - params.gapOpen), (short)(curE[j - 1] - params.gapExtend));
            curF[j] = max_s((short)(prevH[j] - params.gapOpen), (short)(prevF[j] - params.gapExtend));
            short h = max_s((short)0, matchScore);
            h = max_s(h, curE[j]);
            h = max_s(h, curF[j]);
            curH[j] = h;
            best = max_s(best, h);
            diag = prevH[j];
        }

        for (uint j = 0; j <= queryLen; ++j) {
            prevH[j] = curH[j];
            prevF[j] = curF[j];
        }
    }

    outScores[tid] = (int)best;
}

kernel void smith_waterman_score_int8(
    constant Params &params [[buffer(0)]],
    constant short *profile [[buffer(1)]],
    constant char *residueIndex [[buffer(2)]],
    device const char *dbSeqs [[buffer(3)]],
    device const uint *dbOffsets [[buffer(4)]],
    device const uint *dbLengths [[buffer(5)]],
    device int *outScores [[buffer(6)]],
    device uchar *outOverflow [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.dbCount) return;

    const uint dbLen = dbLengths[tid];
    const uint dbOffset = dbOffsets[tid];
    const uint queryLen = min(params.queryLen, (uint)MAX_QUERY_LEN);

    thread char prevH[MAX_QUERY_LEN + 1];
    thread char prevF[MAX_QUERY_LEN + 1];
    thread char curH[MAX_QUERY_LEN + 1];
    thread char curE[MAX_QUERY_LEN + 1];
    thread char curF[MAX_QUERY_LEN + 1];

    for (uint j = 0; j <= queryLen; ++j) {
        prevH[j] = 0;
        prevF[j] = 0;
    }

    bool overflow = false;
    char best = 0;
    for (uint i = 1; i <= dbLen; ++i) {
        curH[0] = 0;
        curE[0] = 0;
        curF[0] = 0;
        int diag = prevH[0];
        const char dbResidue = dbSeqs[dbOffset + i - 1];

        for (uint j = 1; j <= queryLen; ++j) {
            const int matchScore = diag + profileLookup(profile, residueIndex, queryLen, dbResidue, j - 1);
            const int eVal = max(curH[j - 1] - params.gapOpen, curE[j - 1] - params.gapExtend);
            const int fVal = max(prevH[j] - params.gapOpen, prevF[j] - params.gapExtend);
            int h = max(0, matchScore);
            h = max(h, eVal);
            h = max(h, fVal);

            if (h > 127 || eVal > 127 || fVal > 127 || eVal < -128 || fVal < -128) {
                overflow = true;
            }

            curE[j] = (char)clamp(eVal, -128, 127);
            curF[j] = (char)clamp(fVal, -128, 127);
            curH[j] = (char)clamp(h, -128, 127);
            best = max(best, curH[j]);
            diag = prevH[j];
        }

        for (uint j = 0; j <= queryLen; ++j) {
            prevH[j] = curH[j];
            prevF[j] = curF[j];
        }

        if (overflow) break;
    }

    outScores[tid] = (int)best;
    outOverflow[tid] = overflow ? 1 : 0;
}

// Wavefront (anti-diagonal) Smith-Waterman kernel.
// One threadgroup per database sequence. Threadgroup size = 32 (one simdgroup).
// Threads process anti-diagonals k = i+j in parallel. Each thread handles one
// query column j. Requires queryLen ≤ 32.
// 
// Algorithm:
// - Anti-diagonals k = 2 to dbLen + queryLen
// - Thread t handles column j = t+1, row i = k - j
// - H[i,j] = max(0, H[i-1,j-1] + sub, E[i,j], F[i,j])
// - E[i,j] = max(H[i,j-1] - gapOpen, E[i,j-1] - gapExtend)  // LEFT neighbor on SAME k
// - F[i,j] = max(H[i-1,j] - gapOpen, F[i-1,j] - gapExtend)  // UP neighbor on k-1
// - Diagonal H[i-1,j-1] from k-2
//
// Implementation:
// - For each anti-diagonal k:
//   1. All active threads compute diag, F, H_noE in parallel (independent)
//   2. Store intermediate values in threadgroup memory
//   3. Sequential pass: thread t=0,1,2... computes E and H using left neighbor's values
//   4. All threads sync
// - F recurrence: threadgroup memory stores H and F from k-1
// - Diagonal: thread-local H from k-2
// - Three anti-diagonals in flight: k-2 (diag), k-1 (F), k (current)

kernel void smith_waterman_score_wavefront(
    constant Params &params [[buffer(0)]],
    constant short *profile [[buffer(1)]],
    constant char *residueIndex [[buffer(2)]],
    device const char *dbSeqs [[buffer(3)]],
    device const uint *dbOffsets [[buffer(4)]],
    device const uint *dbLengths [[buffer(5)]],
    device int *outScores [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (tid >= params.dbCount) return;

    const uint dbLen = dbLengths[tid];
    const uint dbOffset = dbOffsets[tid];
    const uint queryLen = params.queryLen;

    if (queryLen > WAVEFRONT_THREADS) {
        if (lane == 0) outScores[tid] = -1;
        return;
    }

    const short gapOpen = params.gapOpen;
    const short gapExtend = params.gapExtend;

    short best = 0;

    // Threadgroup memory: H and F for anti-diagonal k-1 (for F recurrence)
    // [2][WAVEFRONT_THREADS] for ping-pong between k-1 and k
    threadgroup short tg_H_k1[2][WAVEFRONT_THREADS];
    threadgroup short tg_F_k1[2][WAVEFRONT_THREADS];

    // Threadgroup memory: current anti-diagonal's H and E (for E recurrence)
    threadgroup short tg_H_curr[WAVEFRONT_THREADS];
    threadgroup short tg_E_curr[WAVEFRONT_THREADS];

    // Threadgroup memory: intermediate values for sequential E pass
    threadgroup short tg_diag[WAVEFRONT_THREADS];
    threadgroup short tg_f_val[WAVEFRONT_THREADS];
    threadgroup short tg_H_noE[WAVEFRONT_THREADS];

    // Thread-local: H for anti-diagonal k-2 (for diagonal)
    thread short H_k2[WAVEFRONT_THREADS];

    // Initialize
    tg_H_k1[0][lane] = 0; tg_F_k1[0][lane] = 0;
    tg_H_k1[1][lane] = 0; tg_F_k1[1][lane] = 0;
    tg_H_curr[lane] = 0;
    tg_E_curr[lane] = 0;
    tg_diag[lane] = 0;
    tg_f_val[lane] = 0;
    tg_H_noE[lane] = 0;
    H_k2[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Total anti-diagonals: k = 2 to dbLen + queryLen
    const uint k_min = 2;
    const uint k_max = dbLen + queryLen;

    for (uint k = k_min; k <= k_max; ++k) {
        // Valid j range for this anti-diagonal
        const uint j_min = max(1u, k <= dbLen ? 1u : k - dbLen);
        const uint j_max = min(queryLen, k - 1);

        const uint t = lane;
        const uint j_col = 1 + t;
        const bool active = (t < queryLen && j_col >= j_min && j_col <= j_max);

        uint i = 0;
        if (active) {
            i = k - j_col;
        }

        const uint ping = k & 1;
        const uint pong = 1 - ping;

        short diag = 0, f_val = 0, H_noE = 0;
        char dbResidue = 0;
        short rIdx = 0, subScore = 0;

        // Phase 1: All active threads compute diag, F, H_noE in parallel
        if (active) {
            uint j = 1 + lane;
            i = k - j;

            const char dbResidue = dbSeqs[dbOffset + i - 1];
            const short rIdx = residueIndex[(uchar)dbResidue];
            const short subScore = profile[rIdx * params.queryLen + (j - 1)];

            // Diagonal: H[i-1,j-1] from k-2
            diag = H_k2[lane] + subScore;

            // F recurrence: max(H[i-1,j] - gapOpen, F[i-1,j] - gapExtend)
            f_val = max((short)(tg_H_k1[ping][lane] - gapOpen),
                         (short)(tg_F_k1[ping][lane] - gapExtend));

            // H without E: max(0, diag, F)
            H_noE = (short)0;
            if (diag > H_noE) H_noE = diag;
            if (f_val > H_noE) H_noE = f_val;

            // Store in threadgroup memory for sequential pass
            tg_diag[lane] = diag;
            tg_f_val[lane] = f_val;
            tg_H_noE[lane] = H_noE;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: Sequential E computation left-to-right
        // Thread 0 computes first, then thread 1, etc.
        for (uint t = 0; t < queryLen; ++t) {
            // Check if this thread is active for this anti-diagonal
            const uint t_j = t + 1;
            const uint t_i = k - t_j;
            const bool t_active = (t < queryLen && t_j >= j_min && t_j <= j_max);

            if (lane == t && t_active) {
                // This thread's turn to compute
                short E_val = 0;

                if (t == 0) {
                    // Leftmost column: boundary
                    E_val = (short)(0 - gapOpen);
                    short alt = (short)(0 - gapExtend);
                    if (alt > E_val) E_val = alt;
                } else {
                    // Read left neighbor's H and E
                    short H_left = tg_H_curr[t - 1];
                    short E_left = tg_E_curr[t - 1];
                    E_val = H_left - gapOpen;
                    short alt = E_left - gapExtend;
                    if (alt > E_val) E_val = alt;
                }

                // Get stored intermediate values
                diag = tg_diag[t];
                f_val = tg_f_val[t];

                // H = max(0, diag, E, F)
                short h = (short)0;
                if (diag > h) h = diag;
                if (E_val > h) h = E_val;
                if (f_val > h) h = f_val;

                // Update best score
                if (h > best) best = h;

                // Write current H and E for right neighbor
                tg_H_curr[t] = h;
                tg_E_curr[t] = E_val;

                // Update thread-local H for diagonal (k+2 will need this as k-2)
                H_k2[t] = h;

                // Store current H and F for next anti-diagonal's F (k+1)
                const uint ping = k & 1;
                const uint pong = 1 - ping;
                tg_H_k1[pong][t] = h;
                tg_F_k1[pong][t] = tg_f_val[t];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Clear threadgroup memory for next anti-diagonal
        if (lane < queryLen) {
            tg_H_curr[lane] = 0;
            tg_E_curr[lane] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce best across threadgroup
    short tg_best = best;
    for (uint offset = 16; offset > 0; offset >>= 1) {
        short other = simd_shuffle_down(tg_best, offset);
        tg_best = tg_best > other ? tg_best : other;
    }
    if (lane == 0) {
        outScores[tid] = (int)tg_best;
    }
}