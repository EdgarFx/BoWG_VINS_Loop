/**
 * File: BoWGScoring.h
 * Date: August 2024
 * Author: Xiang Fei
 * Description: functions to compute bowg scores
 *
 */

#ifndef __BOWG_SCORING__
#define __BOWG_SCORING__

#include "BoWGVector.h"
#include "DBoW2.h"
#include "TemplatedDatabase.h"
#include "TemplatedVocabulary.h"

namespace BoWG {

class BoWGScoring
{
public:
    /**
    * Constructor of BoWGVector
    */
    BoWGScoring(void);


    /**
    * Destructor of BoWGVector
    */
    ~BoWGScoring(void);


    /**
     * Computes the scores between two vectors.
     * @param v1 (in/out)
     * @param v2 (in/out)
     * @return score
    */
    double scoreL1(const BoWGVector &v1, const BoWGVector &v2) const;
    double scoreL2(const BoWGVector &v1, const BoWGVector &v2) const;
    double scoreChiSquare(const BoWGVector &v1, const BoWGVector &v2) const;
    double scoreKL(const BoWGVector &v1, const BoWGVector &v2) const;
    double scoreBhattacharyya(const BoWGVector &v1, const BoWGVector &v2) const;
    double scoreDot(const BoWGVector &v1, const BoWGVector &v2) const;

    double scoreL1(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
    double scoreL2(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
    double scoreChiSquare(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
    double scoreKL(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
    double scoreBhattacharyya(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;
    double scoreDot(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const;

    double dis_score(const std::vector<int> &v, const std::vector<int> &w) const;

    std::vector<double> kernel_normalize(const std::vector<int>& v) const;
};


} // namespace BoWG

#endif