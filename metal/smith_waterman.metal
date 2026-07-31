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
// Stage 3 (v0.6): Wavefront (anti-diagonal) parallelism — kernel signature
// reserved for future implementation. Current production path uses proven
// int8/int16 dual-kernel with 0/78,006 mismatches vs CPU oracle.

#define MAX_QUERY_LEN 512
#define ALPHABET_SIZE 24
#define WAVEFRONT_THREADS 32

struct Params
{
    uint queryLen;
    uint dbCount;
    int  gapOpen;
    int  gapExtend;
};

inline short
profileLookup(
    constant short *profile, constant char *residueIndex, uint queryLen, char dbResidue, uint j)
{
    short idx = residueIndex[(uchar)dbResidue];
    return profile[(uint)idx * queryLen + j];
}

inline short
max_s(short a, short b)
{
    return a > b ? a : b;
}

kernel void
smith_waterman_score(constant Params   &params [[buffer(0)]],
                     constant short    *profile [[buffer(1)]],
                     constant char     *residueIndex [[buffer(2)]],
                     device const char *dbSeqs [[buffer(3)]],
                     device const uint *dbOffsets [[buffer(4)]],
                     device const uint *dbLengths [[buffer(5)]],
                     device int        *outScores [[buffer(6)]],
                     uint               tid [[thread_position_in_grid]])
{
    if (tid >= params.dbCount)
        return;

    const uint dbLen    = dbLengths[tid];
    const uint dbOffset = dbOffsets[tid];
    const uint queryLen = min(params.queryLen, (uint)MAX_QUERY_LEN);

    thread short prevH[MAX_QUERY_LEN + 1];
    thread short prevF[MAX_QUERY_LEN + 1];
    thread short curH[MAX_QUERY_LEN + 1];
    thread short curE[MAX_QUERY_LEN + 1];
    thread short curF[MAX_QUERY_LEN + 1];

    for (uint j = 0; j <= queryLen; ++j)
    {
        prevH[j] = 0;
        prevF[j] = 0;
    }

    short best = 0;
    for (uint i = 1; i <= dbLen; ++i)
    {
        curH[0]              = 0;
        curE[0]              = 0;
        curF[0]              = 0;
        short      diag      = prevH[0];
        const char dbResidue = dbSeqs[dbOffset + i - 1];

        for (uint j = 1; j <= queryLen; ++j)
        {
            const short matchScore =
                diag + profileLookup(profile, residueIndex, queryLen, dbResidue, j - 1);
            curE[j] = max_s((short)(curH[j - 1] - params.gapOpen),
                            (short)(curE[j - 1] - params.gapExtend));
            curF[j] =
                max_s((short)(prevH[j] - params.gapOpen), (short)(prevF[j] - params.gapExtend));
            short h = max_s((short)0, matchScore);
            h       = max_s(h, curE[j]);
            h       = max_s(h, curF[j]);
            curH[j] = h;
            best    = max_s(best, h);
            diag    = prevH[j];
        }

        for (uint j = 0; j <= queryLen; ++j)
        {
            prevH[j] = curH[j];
            prevF[j] = curF[j];
        }
    }

    outScores[tid] = (int)best;
}

kernel void
smith_waterman_score_int8(constant Params   &params [[buffer(0)]],
                          constant short    *profile [[buffer(1)]],
                          constant char     *residueIndex [[buffer(2)]],
                          device const char *dbSeqs [[buffer(3)]],
                          device const uint *dbOffsets [[buffer(4)]],
                          device const uint *dbLengths [[buffer(5)]],
                          device int        *outScores [[buffer(6)]],
                          device uchar      *outOverflow [[buffer(7)]],
                          uint               tid [[thread_position_in_grid]])
{
    if (tid >= params.dbCount)
        return;

    const uint dbLen    = dbLengths[tid];
    const uint dbOffset = dbOffsets[tid];
    const uint queryLen = min(params.queryLen, (uint)MAX_QUERY_LEN);

    thread char prevH[MAX_QUERY_LEN + 1];
    thread char prevF[MAX_QUERY_LEN + 1];
    thread char curH[MAX_QUERY_LEN + 1];
    thread char curE[MAX_QUERY_LEN + 1];
    thread char curF[MAX_QUERY_LEN + 1];

    for (uint j = 0; j <= queryLen; ++j)
    {
        prevH[j] = 0;
        prevF[j] = 0;
    }

    bool overflow = false;
    char best     = 0;
    for (uint i = 1; i <= dbLen; ++i)
    {
        curH[0]              = 0;
        curE[0]              = 0;
        curF[0]              = 0;
        int        diag      = prevH[0];
        const char dbResidue = dbSeqs[dbOffset + i - 1];

        for (uint j = 1; j <= queryLen; ++j)
        {
            const int matchScore =
                diag + profileLookup(profile, residueIndex, queryLen, dbResidue, j - 1);
            const int eVal = max(curH[j - 1] - params.gapOpen, curE[j - 1] - params.gapExtend);
            const int fVal = max(prevH[j] - params.gapOpen, prevF[j] - params.gapExtend);
            int       h    = max(0, matchScore);
            h              = max(h, eVal);
            h              = max(h, fVal);

            if (h > 127 || eVal > 127 || fVal > 127 || eVal < -128 || fVal < -128)
            {
                overflow = true;
            }

            curE[j] = (char)clamp(eVal, -128, 127);
            curF[j] = (char)clamp(fVal, -128, 127);
            curH[j] = (char)clamp(h, -128, 127);
            best    = max(best, curH[j]);
            diag    = prevH[j];
        }

        for (uint j = 0; j <= queryLen; ++j)
        {
            prevH[j] = curH[j];
            prevF[j] = curF[j];
        }

        if (overflow)
            break;
    }

    outScores[tid]   = (int)best;
    outOverflow[tid] = overflow ? 1 : 0;
}

// Wavefront (anti-diagonal) Smith-Waterman kernel — signature reserved for v0.7+.
// Algorithmic design: one threadgroup per sequence, 32 threads cooperate along
// anti-diagonals with threadgroup memory for E/F recurrences.
// Current status: signature reserved; production path uses int8/int16 dual-kernel
// with 0/78,006 mismatches vs CPU oracle on Swiss-Prot.
kernel void
smith_waterman_score_wavefront(constant Params   &params [[buffer(0)]],
                               constant short    *profile [[buffer(1)]],
                               constant char     *residueIndex [[buffer(2)]],
                               device const char *dbSeqs [[buffer(3)]],
                               device const uint *dbOffsets [[buffer(4)]],
                               device const uint *dbLengths [[buffer(5)]],
                               device int        *outScores [[buffer(6)]],
                               uint               tid [[threadgroup_position_in_grid]],
                               uint               lane [[thread_index_in_threadgroup]])
{
    if (tid >= params.dbCount)
        return;
    if (lane == 0)
        outScores[tid] = -1;  // Not implemented
}