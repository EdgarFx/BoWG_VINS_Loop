/**
 * File: BoWGDatabase.h
 * Date: August 2024
 * Author: Xiang Fei
 * Description: DoWG database of images
 *
 */

#ifndef __BOWG_DATABASE__
#define __BOWG_DATABASE__

#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <list>
#include <set>
#include <map>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "DBoW2.h"
#include "TemplatedDatabase.h"
#include "TemplatedVocabulary.h"

#include "BoWGVector.h"
#include "BoWGScoring.h"

namespace BoWG {

// Id of items of the database (image id)
typedef unsigned int ItemID;

class BoWGDatabase
{
public:

    /**
    * Constructor of BoWGDatabase
    * @param wg_weight_type weighting type of word group
    * @param wg_scoring_type scoring type of word group
    * @param w_scoring_type scoring type of word (for computing the normalized score)
    */
    BoWGDatabase(int wg_weight_type = 0, int wg_scoring_type = 1, int w_scoring_type = 0);

    /**
    * Destructor of BoWGDatabase
    */
    ~BoWGDatabase(void);

    /**
    * check whether the current word groups is in the database or not
    * if yes, return the word group id, otherwise, add the word group into our database and assign its id
    * @param wid1 center word id of the consider word groups
    */ 
    WordGroupID queryWordGroup(DBoW2::WordId wid1);

    /**
    * add new world group to the database and return its id
    * @param wg_keys center word id of the consider word groups
    */ 
    WordGroupID db_add(DBoW2::WordId wg_key);

    /**
    * obtain the BoWGVector for an image
    * @param wg_ids word group ids of an image
    */ 
    BoWGVector computeBoWGVector(std::vector<WordGroupID> wg_ids);

    /**
    * update the BoWGVector Table and Inverse Index Table
    * @param bowgVector BoWGVector of the image
    * @param entry_id Id of the image
    */ 
    void add(BoWGVector& bowgVector, ItemID entry_id);

    /**
    * update the Distribution Table
    * @param vec distribution vector of the image
    */ 
    void dist_add(std::vector<int>& vec);

    // obtain the image id
    ItemID get_itemId(void);

    /**
    * query the word group results using inverse index table
    * @param vec BoWGVector of the query image
    * @param ret original word group query results
    * @param wg_item_vec a vector for later word/word group results alignment
    * @param max_results max number of results
    * @param max_id the max id of images
    */ 
    void wg_queryL1(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
    void wg_queryL2(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
    void wg_queryCHI_SQUARE(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
    void wg_queryKL(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
    void wg_queryBHATTACHARYYA(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
    void wg_queryDOT(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);

    /**
    * query the final results using normalized and temporal word score
    * @param ret original word query results
    * @param out_ret output results
    * @param bowVector bowVector of the query image
    * @param max_results max number of results
    * @param use_temporal_score whether to use temporal score
    * @param prev_weight_th the maximum previou score weight
    * @param temporal_param temporal score parameter to compute the weight
    */ 
    void query_words(DBoW2::QueryResults &ret, DBoW2::QueryResults &out_ret, DBoW2::BowVector &bowVector, int max_results, 
                        bool use_temporal_score=true, double prev_weight_th = 0.5, double temporal_param=1.0);

    /**
    * query the final results using normalized and temporal word score and word group score
    * @param ret original word query results
    * @param out_ret output results
    * @param bowVector bowVector of the query image
    * @param bowgVector bowgVector of the query image
    * @param max_results max number of results
    * @param max_id the max id of images
    * @param w_weight the weight of word score for combined score
    * @param use_temporal_score whether to use temporal score
    * @param prev_weight_th the maximum previou score weight
    * @param temporal_param temporal score parameter to compute the weight
    */ 
    void query_bowg(DBoW2::QueryResults &ret, DBoW2::QueryResults &out_ret, DBoW2::BowVector &bowVector, BoWGVector &bowgVector, int max_results = 1, 
            int max_id = -1, double w_weight=0.7, bool use_temporal_score=true, double prev_weight_th = 0.5, double temporal_param=1.0);

    
    /**
    * query the final results using normalized and temporal word score, word group score, and distribution score
    * @param ret original word query results
    * @param out_ret output results
    * @param bowVector bowVector of the query image
    * @param bowgVector bowgVector of the query image
    * @param dist_vec distribution vector of the query image
    * @param max_results max number of results
    * @param max_id the max id of images
    * @param w_weight the weight of word score for combined score
    * @param wg_weight the weight of word group score for combined score
    * @param use_temporal_score whether to use temporal score
    * @param prev_weight_th the maximum previou score weight
    * @param temporal_param temporal score parameter to compute the weight
    */ 
    void query_bowg(DBoW2::QueryResults &ret, DBoW2::QueryResults &out_ret, DBoW2::BowVector &bowVector, BoWGVector &bowgVector, std::vector<int> dist_vec, int max_results = 1, 
        int max_id = -1, double w_weight=0.35, double wg_weight=0.35, bool use_temporal_score=true, double prev_weight_th = 0.5, double temporal_param=1.0);

    /**
    * compute the islands
    * @param ret query results
    * @param groupMatches islands
    * @param candidates_th similarity threshold
    * @param max_intraisland_gap max separation between matches to consider them of the same island
    */ 
    bool matchGrouping(DBoW2::QueryResults &ret, std::vector<std::vector<int>> &groupMatches, double candidates_th, int max_intraisland_gap);

    // obtain the matched island index
    int islandMatching(DBoW2::QueryResults &ret, std::vector<std::vector<int>> &groupMatches);

    // temporal consistency check, consider previous k islands
    bool temporalConsistency(DBoW2::QueryResults &ret, std::vector<int> &island, int k, int max_group_gap, int max_query_gap);

    // get the number of word groups
    int get_num_wg(void);

    /*Compute the input of kernel density estimation
     *input is the keypoints of the current keyframe */ 
    std::vector<int> kernel_input(std::vector<cv::KeyPoint> keypoints, cv::Point2f center_pt, int batch);

    // stores the last time query result
    void store_last_res(DBoW2::QueryResults &ret);

    // last time query result, used to compute our proposed temporal consistency score
    std::map<int, double> last_query_res;

    // stores the information of matched islands in timesteps
    std::map<int, std::vector<int>> Islands_map;

    // stores the query results in timesteps
    std::map<int, DBoW2::QueryResults> res_table;

    // previous bow vector, used for normalization
    DBoW2::BowVector prev_bow_vec;

    // previous bowg vector, used for normalization
    BoWGVector prev_bowg_vec;

    // previous bowg vector, used for normalization
    std::vector<int> prev_dist_vec;

    // Scoring
    BoWGScoring bowg_scoring;

    int cur_image_id = 0;

    double min_prev_score = 0.005;
    double min_prev_wg_score = 0.005;
    double min_prev_dist_score = 0.003;

protected:

    /// Item of IFRow
    struct GroupIFPair
    {
        /// Item id
        ItemID item_id;
        
        /// Word group weight in this item
        WordGroupValue wg_weight;
        
        /**
        * Creates an empty pair
        */
        GroupIFPair(){}
        
        /**
        * Creates an inverted file pair
        * @param eid item id
        * @param wv word group weight
        */
        GroupIFPair(ItemID eid, WordGroupValue wv): item_id(eid), wg_weight(wv) {}
        
        /**
        * Compares the item ids
        * @param eid
        * @return true iff this item id is the same as eid
        */
        inline bool operator==(ItemID eid) const { return item_id == eid; }
    };

    /// Row of InvertedFile
    typedef std::list<GroupIFPair> GroupIFRow;
    // GroupIFRow are sorted in ascending item id order

    /// Inverted index
    typedef std::map<WordGroupID,GroupIFRow> GroupInvertedFile; 
    // GroupInvertedFile[wg_id] --> inverted file of that word group

    /* BoWGVector table declaration*/
    typedef std::vector<BoWGVector> BoWGVectorTable;
    // BoWGVectorTable[item_id] --> the BoWGVector of an image, only useful when don't want to use inverse index table to compute scores

    // Distribution table decalaration
    typedef std::vector<std::vector<int>> DisTable;
    // DisTable[item_id] --> distribution vector of an image

    // IDF Table
    typedef std::map<WordGroupID,int> IDFTable;

protected:
    // vocabulary (BoWG Table) wg id and the number of the wg
    std::map<WordGroupID, int> voc_wg;

    // inverted index table
    GroupInvertedFile m_ifile;

    // BoWGVector table
    BoWGVectorTable m_dtable;

    // distribution table
    DisTable dis_table;

    // total word groups (consider repeat)
    int numOfWg;

    int w_scoring_type; // to compute the word normalization score
    int wg_weight_type;
    int wg_scoring_type;
};

} // namespace BoWG


#endif