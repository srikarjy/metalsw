#include "parasail_score.hpp"

#ifdef METALSW_USE_PARASAIL

#    include <parasail.h>
#    include <parasail/matrices/blosum62.h>

namespace metalsw
{

int
parasailScore(const std::string &query, const std::string &db, int gapOpen, int gapExtend)
{
    parasail_result_t *result = parasail_sw(query.c_str(),
                                            static_cast<int>(query.size()),
                                            db.c_str(),
                                            static_cast<int>(db.size()),
                                            gapOpen,
                                            gapExtend,
                                            &parasail_blosum62);
    int                score  = result->score;
    parasail_result_free(result);
    return score;
}

}  // namespace metalsw

#endif  // METALSW_USE_PARASAIL
