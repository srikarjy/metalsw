#include <metal_stdlib>
using namespace metal;

// Naive inter-sequence Smith-Waterman (Gotoh affine gap), one thread per
// database sequence. This is a direct MSL port of sw_reference.cpp's
// linear-space recurrence — no query profile, no INT8 packing, no
// threadgroup tuning. Those are Stage 2 (v0.3) work.

#define MAX_QUERY_LEN 512
#define ALPHABET_SIZE 24

struct Params {
    uint queryLen;
    uint dbCount;
    int gapOpen;
    int gapExtend;
};

inline short blosumLookup(constant char *residueIndex,
                           constant short *blosum,
                           char a, char b) {
    short ia = residueIndex[(uchar)a];
    short ib = residueIndex[(uchar)b];
    return blosum[ia * ALPHABET_SIZE + ib];
}

kernel void smith_waterman_score(
    constant char *query [[buffer(0)]],
    constant Params &params [[buffer(1)]],
    constant short *blosum [[buffer(2)]],
    constant char *residueIndex [[buffer(3)]],
    device const char *dbSeqs [[buffer(4)]],
    device const uint *dbOffsets [[buffer(5)]],
    device const uint *dbLengths [[buffer(6)]],
    device int *outScores [[buffer(7)]],
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
            const short matchScore = diag + blosumLookup(residueIndex, blosum, dbResidue, query[j - 1]);
            curE[j] = max((short)(curH[j - 1] - params.gapOpen), (short)(curE[j - 1] - params.gapExtend));
            curF[j] = max((short)(prevH[j] - params.gapOpen), (short)(prevF[j] - params.gapExtend));
            short h = max((short)0, matchScore);
            h = max(h, curE[j]);
            h = max(h, curF[j]);
            curH[j] = h;
            best = max(best, h);
            diag = prevH[j];
        }

        for (uint j = 0; j <= queryLen; ++j) {
            prevH[j] = curH[j];
            prevF[j] = curF[j];
        }
    }

    outScores[tid] = (int)best;
}
