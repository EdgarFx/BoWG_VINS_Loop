/**
 * File: Parameters.cpp
 * Date: Dec 2024
 * Author: Xiang Fei
 * Description: parameters used by BoWG
 */

#include "Parameters.h"
#include <opencv2/core/core.hpp>

// std::string IMAGE_PATH;
// std::string RESULTS_PATH;
// std::string GT_PATH;
// std::string BRIEF_PATTERN_FILE;
// std::string VOCABULARY_PATH;
// bool GT_USE;

int DISLOCAL;
int FEATURE_CNT;
bool MS_FEATURE;
int MS_LEVELS;
float FEATURE_SIZE;
int W_WEIGHT_TYPE;
int W_SCORING_TYPE;
int WG_WEIGHT_TYPE;
int WG_SCORING_TYPE;
bool USE_WG;
bool USE_DISTRIBUTION;
int DISTRIBUTION_BATCH;
double W_WEIGHT;
double WG_WEIGHT;
bool USE_TEMPORAL_SCORE;
double PREV_WEIGHT;
bool PREV_LARGER_FLAG;
double SIMILARITY_TH;
int TEMPORAL_K;
int MAX_INTRAISLAND_GAP;
int MAX_DISTANCE_BETWEEN_GROUPS;
int MAX_DISTANCE_BETWEEN_QUERIES;
bool USE_GEOM;
bool GEOM_DI;
int DI_LEVEL;
int MIN_FPOINTS;
double MAX_REPROJECTION_ERROR;
double RANSAC_PROBABILITY;
int MAX_RANSAC_ITERATIONS;
double MAX_NEIGHBOR_RATIO;
double MIN_PREV_W_SCORE;
double MIN_PREV_WG_SCORE;
double MIN_PREV_DIST_SCORE;


bool Parameters::readParameters(const std::string& config_file) {
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Failed to open settings file: " << config_file << std::endl;
        return false;
    }

    // readValue(fsSettings, "image_path", IMAGE_PATH);
    // readValue(fsSettings, "results_path", RESULTS_PATH, "../CorMatrix.txt");
    readValue(fsSettings, "w_weight", W_WEIGHT, 0.7);
    readValue(fsSettings, "wg_weight", WG_WEIGHT, 0.3);
    readValue(fsSettings, "similarity_th", SIMILARITY_TH, 0.4);
    readValue(fsSettings, "temporal_k", TEMPORAL_K, 1);

    // readValue(fsSettings, "gt_use", GT_USE, false);
    // readValue(fsSettings, "gt_path", GT_PATH, "../GT/GT_loop.txt");
    // readValue(fsSettings, "BRIEF_PATTERN_FILE", BRIEF_PATTERN_FILE, "../support_files/brief_pattern.yml");
    // readValue(fsSettings, "vocabulary_path", VOCABULARY_PATH, "../support_files/bovisa_brief_k10L6.bin");
    readValue(fsSettings, "dislocal", DISLOCAL, 70);
    readValue(fsSettings, "feature_cnt", FEATURE_CNT, 300);
    readValue(fsSettings, "ms_feature", MS_FEATURE, true);
    readValue(fsSettings, "feature_size", FEATURE_SIZE, 5.0f);
    readValue(fsSettings, "ms_levels", MS_LEVELS, 5);
    readValue(fsSettings, "w_weight_type", W_WEIGHT_TYPE, 0);
    readValue(fsSettings, "wg_weight_type", WG_WEIGHT_TYPE, 0);
    readValue(fsSettings, "w_scoring_type", W_SCORING_TYPE, 0);
    readValue(fsSettings, "wg_scoring_type", WG_SCORING_TYPE, 1);
    readValue(fsSettings, "use_wg", USE_WG, true);
    readValue(fsSettings, "use_distribution", USE_DISTRIBUTION, false);
    readValue(fsSettings, "distribution_batch", DISTRIBUTION_BATCH, 36);
    readValue(fsSettings, "use_temporal_score", USE_TEMPORAL_SCORE, true);
    readValue(fsSettings, "prev_weight", PREV_WEIGHT, 0.2);
    readValue(fsSettings, "prev_larger_flag", PREV_LARGER_FLAG, false);
    readValue(fsSettings, "max_intraisland_gap", MAX_INTRAISLAND_GAP, 3);
    readValue(fsSettings, "max_distance_between_groups", MAX_DISTANCE_BETWEEN_GROUPS, 1);
    readValue(fsSettings, "max_distance_between_queries", MAX_DISTANCE_BETWEEN_QUERIES, 1);
    readValue(fsSettings, "use_geom", USE_GEOM, true);
    readValue(fsSettings, "GEOM_DI", GEOM_DI, true);
    readValue(fsSettings, "di_level", DI_LEVEL, 5);
    readValue(fsSettings, "min_Fpoints", MIN_FPOINTS, 30);
    readValue(fsSettings, "max_reprojection_error", MAX_REPROJECTION_ERROR, 7.0);
    readValue(fsSettings, "ransac_probability", RANSAC_PROBABILITY, 0.9);
    readValue(fsSettings, "max_ransac_iterations", MAX_RANSAC_ITERATIONS, 200);
    readValue(fsSettings, "max_neighbor_ratio", MAX_NEIGHBOR_RATIO, 0.8);
    readValue(fsSettings, "min_prev_w_score", MIN_PREV_W_SCORE, 0.005);
    readValue(fsSettings, "min_prev_wg_score", MIN_PREV_WG_SCORE, 0.005);
    readValue(fsSettings, "min_prev_dist_score", MIN_PREV_DIST_SCORE, 0.003);

    fsSettings.release();
    return true;
}

// example use
/* 
int main() {
    if (!Parameters::readParameters("config.yaml")) {
        std::cerr << "Failed to read parameters" << std::endl;
        return -1;
    }

    std::cout << "Similarity Threshold: " << SIMILARITY_TH << std::endl;
    std::cout << "Max features: " << FEATURE_CNT << std::endl;

    return 0;
}
*/