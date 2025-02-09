/**
 * File: GeometricUtils.h
 * Date: Dec 2024
 * Author: Xiang Fei
 * Description: Geometrical checking in loop closure detection
*/

#include <vector>
#include <numeric>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "TemplatedVocabulary.h"
#include "TemplatedDatabase.h"
#include "QueryResults.h"
#include "BowVector.h"
#include "DUtils.h"
#include "DVision.h"

using namespace std;
using namespace DUtils;
using namespace DBoW2;
using namespace DVision;

// geometrical consistency checking with BF
bool isGeometricallyConsistent_Exhaustive(
    const std::vector<BRIEF::bitset>& descriptors1,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<BRIEF::bitset>& descriptors2,
    const std::vector<cv::KeyPoint>& keypoints2,
    std::vector<cv::DMatch>& inlierMatches,
    float inlierTh = 30,
    float inlierRatioThreshold = 0.2
);

// geometrical consistency checking with direct table
template<class TDescriptor, class F>
bool isGeometricallyConsistent_DI(DVision::FSolver m_fsolver, BriefDatabase &m_database, 
    vector<vector<TDescriptor>> &m_image_descriptors, vector<vector<cv::KeyPoint>> &m_image_keys, EntryId old_entry, 
    const std::vector<cv::KeyPoint> &keys, const std::vector<TDescriptor> &descriptors, 
    const FeatureVector &curvec, vector<cv::Point2f> &old_points, vector<cv::Point2f> &cur_points,
    vector<uchar> &old_status, vector<uchar> &cur_status,
    int min_Fpoints = 30, double max_reprojection_error = 7.0, 
    double ransac_probability = 0.9, int max_ransac_iterations = 200, double max_neighbor_ratio = 0.8);


template<class TDescriptor, class F>
void getMatches_neighratio(const std::vector<TDescriptor> &A, 
    const vector<unsigned int> &i_A, const vector<TDescriptor> &B,
    const vector<unsigned int> &i_B,
    vector<unsigned int> &i_match_A, vector<unsigned int> &i_match_B, double max_neighbor_ratio);


bool isGeometricallyConsistent_Exhaustive(
    const std::vector<BRIEF::bitset>& descriptors1,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<BRIEF::bitset>& descriptors2,
    const std::vector<cv::KeyPoint>& keypoints2,
    std::vector<cv::DMatch>& inlierMatches,
    float inlierTh,
    float inlierRatioThreshold
) {
    if (descriptors1.empty() || descriptors2.empty() || keypoints1.empty() || keypoints2.empty()) {
        std::cerr << "Empty keypoints or descriptors" << std::endl;
        return false;
    }

    cv::Mat descMat1(descriptors1.size(), 32, CV_8U);
    cv::Mat descMat2(descriptors2.size(), 32, CV_8U);

    for (size_t i = 0; i < descriptors1.size(); ++i) {
        // Create a string representation of the bitset
        std::string bitsetStr;
        for (size_t bitIdx = 0; bitIdx < descriptors1[i].size(); ++bitIdx) {
            bitsetStr += descriptors1[i][bitIdx] ? '1' : '0';  // Append '1' or '0' for each bit
        }

        for (int j = 0; j < 32; ++j) {
            // Convert 8 bits at a time into a uchar
            unsigned long chunk = std::bitset<64>(bitsetStr.substr(j * 8, 8)).to_ulong();
            descMat1.at<uchar>(i, j) = static_cast<uchar>(chunk & 0xFF);
        }
    }

    for (size_t i = 0; i < descriptors2.size(); ++i) {
        std::string bitsetStr;
        for (size_t bitIdx = 0; bitIdx < descriptors2[i].size(); ++bitIdx) {
            bitsetStr += descriptors2[i][bitIdx] ? '1' : '0';
        }

        for (int j = 0; j < 32; ++j) {
            unsigned long chunk = std::bitset<64>(bitsetStr.substr(j * 8, 8)).to_ulong();
            descMat2.at<uchar>(i, j) = static_cast<uchar>(chunk & 0xFF);
        }
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descMat1, descMat2, matches);

    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });

    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    std::vector<unsigned char> inliersMask;
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, 5.0, inliersMask);

    if (H.empty()) {
        std::cerr << "RANSAC Fail" << std::endl;
        return false;
    }

    int inliers = std::count(inliersMask.begin(), inliersMask.end(), 1);
    float inlierRatio = static_cast<float>(inliers) / matches.size();

    if (inlierRatio < inlierRatioThreshold || inliers < inlierTh) {
        std::cerr << "Match Fail: small number of inliers" << std::endl;
        return false;
    }

    inlierMatches.clear();
    for (size_t i = 0; i < inliersMask.size(); ++i) {
        if (inliersMask[i]) {
            inlierMatches.push_back(matches[i]);
        }
    }

    return true;
}


template<class TDescriptor, class F>
bool isGeometricallyConsistent_DI(DVision::FSolver m_fsolver, BriefDatabase &m_database, 
  vector<vector<TDescriptor>> &m_image_descriptors, vector<vector<cv::KeyPoint>> &m_image_keys,
  EntryId old_entry, const std::vector<cv::KeyPoint> &keys, const std::vector<TDescriptor> &descriptors, 
  const FeatureVector &bowvec, int min_Fpoints, double max_reprojection_error, 
  double ransac_probability, int max_ransac_iterations, double max_neighbor_ratio)
{
  const FeatureVector &oldvec = m_database.retrieveFeatures(old_entry);

  // for each word in common, get the closest descriptors
  vector<unsigned int> i_old, i_cur;

  FeatureVector::const_iterator old_it, cur_it; 
  const FeatureVector::const_iterator old_end = oldvec.end();
  const FeatureVector::const_iterator cur_end = bowvec.end();
  

  old_it = oldvec.begin();
  cur_it = bowvec.begin();

  while(old_it != old_end && cur_it != cur_end)
  {
    if(old_it->first == cur_it->first)
    {
      // compute matches between 
      // features old_it->second of m_image_keys[old_entry] and
      // features cur_it->second of keys
      vector<unsigned int> i_old_now, i_cur_now;
      
      getMatches_neighratio<DVision::BRIEF::bitset, FBrief>(
        m_image_descriptors[old_entry], old_it->second, 
        descriptors, cur_it->second,  
        i_old_now, i_cur_now, max_neighbor_ratio);
      
      i_old.insert(i_old.end(), i_old_now.begin(), i_old_now.end());
      i_cur.insert(i_cur.end(), i_cur_now.begin(), i_cur_now.end());
      
      // move old_it and cur_it forward
      ++old_it;
      ++cur_it;
    }
    else if(old_it->first < cur_it->first)
    {
      // move old_it forward
      old_it = oldvec.lower_bound(cur_it->first);
      // old_it = (first element >= cur_it.id)
    }
    else
    {
      // move cur_it forward
      cur_it = bowvec.lower_bound(old_it->first);
      // cur_it = (first element >= old_it.id)
    }
  }
  
  // calculate now the fundamental matrix
  if((int)i_old.size() >= min_Fpoints)
  {
    vector<cv::Point2f> old_points, cur_points;
    
    // add matches to the vectors to calculate the fundamental matrix
    vector<unsigned int>::const_iterator oit, cit;
    oit = i_old.begin();
    cit = i_cur.begin();
    
    for(; oit != i_old.end(); ++oit, ++cit)
    {
      const cv::KeyPoint &old_k = m_image_keys[old_entry][*oit];
      const cv::KeyPoint &cur_k = keys[*cit];
      
      old_points.push_back(old_k.pt);
      cur_points.push_back(cur_k.pt);
    }
  
    cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]);
    cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]);
    
    return m_fsolver.checkFundamentalMat(oldMat, curMat, 
      max_reprojection_error, min_Fpoints,
      ransac_probability, max_ransac_iterations);
  }
  
  return false;
}


template<class TDescriptor, class F>
void getMatches_neighratio(
  const vector<TDescriptor> &A, const vector<unsigned int> &i_A,
  const vector<TDescriptor> &B, const vector<unsigned int> &i_B,
  vector<unsigned int> &i_match_A, vector<unsigned int> &i_match_B, double max_neighbor_ratio) 
{
  i_match_A.resize(0);
  i_match_B.resize(0);
  i_match_A.reserve( min(i_A.size(), i_B.size()) );
  i_match_B.reserve( min(i_A.size(), i_B.size()) );
  vector<unsigned int>::const_iterator ait, bit;
  unsigned int i, j;
  i = 0;
  for(ait = i_A.begin(); ait != i_A.end(); ++ait, ++i)
  {
    int best_j_now = -1;
    double best_dist_1 = 1e9;
    double best_dist_2 = 1e9;
    
    j = 0;
    for(bit = i_B.begin(); bit != i_B.end(); ++bit, ++j)
    {
      double d = F::distance(A[*ait], B[*bit]);
            
      // in i
      if(d < best_dist_1)
      {
        best_j_now = j;
        best_dist_2 = best_dist_1;
        best_dist_1 = d;
      }
      else if(d < best_dist_2)
      {
        best_dist_2 = d;
      }
    }
    
    if(best_dist_1 / best_dist_2 <= max_neighbor_ratio)
    {
      unsigned int idx_B = i_B[best_j_now];
      bit = find(i_match_B.begin(), i_match_B.end(), idx_B);
      
      if(bit == i_match_B.end())
      {
        i_match_B.push_back(idx_B);
        i_match_A.push_back(*ait);
      }
      else
      {
        unsigned int idx_A = i_match_A[ bit - i_match_B.begin() ];
        double d = F::distance(A[idx_A], B[idx_B]);
        if(best_dist_1 < d)
        {
          i_match_A[ bit - i_match_B.begin() ] = *ait;
        }
      }
        
    }
  }
}