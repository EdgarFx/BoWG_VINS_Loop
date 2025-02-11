/**
 * File: BoWGDatabase.cpp
 * Date: July 2023
 * Author: August 2024
 * Description: DoWG database of images
 *
 */


#include "BoWGDatabase.h"


namespace BoWG{

unsigned int abs_diff(ItemID a, ItemID b) {
    return (a > b) ? (a - b) : (b - a);
}

BoWGDatabase::BoWGDatabase(int wg_weight_type, int wg_scoring_type, int w_scoring_type) 
        : numOfWg(0), wg_weight_type(wg_weight_type), wg_scoring_type(wg_scoring_type), w_scoring_type(w_scoring_type)
{
    m_dtable.reserve(20000);
}


BoWGDatabase::~BoWGDatabase(void)
{
}

std::vector<int> BoWGDatabase::kernel_input(std::vector<cv::KeyPoint> keypoints, cv::Point2f center_pt, int batch)
{
    std::vector<int> result(batch,0);
    double unit_angle = 2*M_PI / batch;
    for (int i=0; i<(int)keypoints.size(); i++)
    {
        float pt_x = keypoints[i].pt.x;
        float pt_y = keypoints[i].pt.y;
        double dx = fabs(center_pt.x - pt_x);
        double dy = fabs(center_pt.y - pt_y);
        if (dx==0 && dy==0)
            continue;
        double pt_angle = atan(dx/dy);
        if (pt_x<=center_pt.x && pt_y>center_pt.y)
        {
            pt_angle = M_PI - pt_angle;
        }
        else if (pt_x>center_pt.x && pt_y<=center_pt.y)
        {
            pt_angle = 2*M_PI - pt_angle;
        }
        else if (pt_x>center_pt.x && pt_y>center_pt.y)
        {
            pt_angle = M_PI + pt_angle;
        }
        
        int pt_batch = pt_angle / unit_angle;
        result[pt_batch] += 1;
    }
    return result;
}


WordGroupID BoWGDatabase::queryWordGroup(DBoW2::WordId wid1)
{
    WordGroupID wg_key;

    wg_key = wid1;

    if (voc_wg.size() == 0)
    {
        WordGroupID new_id = db_add(wg_key);
        return new_id;
    }
    if (voc_wg.find(wg_key) == voc_wg.end()) // must be a new word group
    {
        WordGroupID new_id = db_add(wg_key);
        return new_id;
    } else {
        WordGroupID wg_id = wg_key;
        voc_wg[wg_id]++;
        numOfWg++;
        return wg_id;
    }
}


WordGroupID BoWGDatabase::db_add(DBoW2::WordId wg_key)
{
    WordGroupID new_id = wg_key;
    // add to vocaburary
    voc_wg.insert({new_id, 1});
    numOfWg++;

    return new_id;
}

BoWGVector BoWGDatabase::computeBoWGVector(std::vector<WordGroupID> wg_ids)
{
    BoWGVector bowg_vec;
    WordGroupID wg_id;
    for (size_t i=0; i < wg_ids.size(); i++)
    {
        wg_id = wg_ids[i];
        WordGroupValue wg_weight;
        if (wg_weight_type == DBoW2::TF_IDF) {
            double idf = std::log(numOfWg/voc_wg[wg_id]);
            int wg_count = std::count(wg_ids.begin(), wg_ids.end(), wg_id);
            double tf = (double)wg_count / wg_ids.size();
            wg_weight = tf * idf;
        }
        else if (wg_weight_type == DBoW2::TF) {
            int wg_count = std::count(wg_ids.begin(), wg_ids.end(), wg_id);
            double tf = (double)wg_count / wg_ids.size();
            wg_weight = tf;
        }
        else {
            double idf = std::log(numOfWg/voc_wg[wg_id]);
            wg_weight = idf;
        }
        bowg_vec.insert({wg_id, wg_weight});
    }
    bowg_vec.normalize();
    return bowg_vec;
}

void BoWGDatabase::add(BoWGVector& bowgVector, ItemID entry_id)
{
    // update BoWGVector table
    m_dtable.push_back(bowgVector);

    BoWGVector::const_iterator vit;
    // update inverse index table
    for(vit = bowgVector.begin(); vit != bowgVector.end(); ++vit)
    {
        const WordGroupID& wg_id = vit->first;
        const WordGroupValue& wg_weight = vit->second;
        
        GroupIFRow& ifrow = m_ifile[wg_id];
        ifrow.push_back(GroupIFPair(entry_id, wg_weight));
    }
}

void BoWGDatabase::dist_add(std::vector<int>& vec)
{
    dis_table.push_back(vec);
}


ItemID BoWGDatabase::get_itemId(void)
{
    return m_dtable.size();
}

bool compare(const DBoW2::Result& a, const DBoW2::Result& b) {
    return a.Score > b.Score;
}

bool id_compare(const DBoW2::Result& a, const DBoW2::Result& b) {
    return a.Id < b.Id;
}

void BoWGDatabase::store_last_res(DBoW2::QueryResults &ret)
{
    for (size_t i = 0; i < ret.size(); i++)
    {
        last_query_res[ret[i].Id] = ret[i].Score;
    }
}


void BoWGDatabase::wg_queryL1(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results, int max_id)
{
    BoWGVector::const_iterator vit;
    typename GroupIFRow::const_iterator rit;
    
    std::map<ItemID, double> pairs;
    std::map<ItemID, double>::iterator pit;
    
    for(vit = vec.begin(); vit != vec.end(); ++vit)
    {
        const WordGroupID wg_id = vit->first;
        const WordGroupValue& qvalue = vit->second;
        
        const GroupIFRow& row = m_ifile[wg_id];
        // GroupIFRow are sorted in ascending entry_id order
        
        for(rit = row.begin(); rit != row.end(); ++rit)
        {
        const ItemID entry_id = rit->item_id;
        const WordGroupValue& dvalue = rit->wg_weight;

        if((int)entry_id < max_id || max_id == -1)
        {
            double value = fabs(qvalue - dvalue) - fabs(qvalue) - fabs(dvalue);
            
            pit = pairs.lower_bound(entry_id);
            if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
            {
            pit->second += value; 
            }
            else
            {
            pairs.insert(pit, 
                std::map<ItemID, double>::value_type(entry_id, value));
            }
        }
        
        } // for each inverted row
    } // for each query word group
        
    // move to vector
    ret.reserve(pairs.size());
    //cit = counters.begin();
    for(pit = pairs.begin(); pit != pairs.end(); ++pit)//, ++cit)
    {
        ret.push_back(DBoW2::Result(pit->first, pit->second));// / cit->second));
        wg_item_vec.push_back(pit->first);
    }
        
    // resulting "scores" are now in [-1 best .. 0 worst]	
    
    // sort vector in ascending order of score
    std::sort(ret.begin(), ret.end());
    // (ret is inverted now --the lower the better--)
    std::sort(wg_item_vec.begin(), wg_item_vec.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results)
        ret.resize(max_results);

    // complete and scale score to [0 worst .. 1 best]
    // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) 
	//		for all i | v_i != 0 and w_i != 0 )
	// (Nister, 2006)
    DBoW2::QueryResults::iterator qit;
    for(qit = ret.begin(); qit != ret.end(); qit++) 
        qit->Score = -qit->Score/2.0;

    std::sort(ret.begin(), ret.end(), id_compare);
}


void BoWGDatabase::wg_queryL2(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results, int max_id)
{
    BoWGVector::const_iterator vit;
    typename GroupIFRow::const_iterator rit;
    
    std::map<ItemID, double> pairs;
    std::map<ItemID, double>::iterator pit;
    
    for(vit = vec.begin(); vit != vec.end(); ++vit)
    {
        const WordGroupID wg_id = vit->first;
        const WordGroupValue& qvalue = vit->second;
        
        const GroupIFRow& row = m_ifile[wg_id];
        // GroupIFRow are sorted in ascending entry_id order
        
        for(rit = row.begin(); rit != row.end(); ++rit)
        {
        const ItemID entry_id = rit->item_id;
        const WordGroupValue& dvalue = rit->wg_weight;

        if((int)entry_id < max_id || max_id == -1)
        {
            double value = - qvalue * dvalue; // minus sign for sorting trick
            
            pit = pairs.lower_bound(entry_id);
            if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
            {
            pit->second += value; 
            }
            else
            {
            pairs.insert(pit, 
                std::map<ItemID, double>::value_type(entry_id, value));
            }
        }
        
        } // for each inverted row
    } // for each query word group
        
    // move to vector
    ret.reserve(pairs.size());
    //cit = counters.begin();
    for(pit = pairs.begin(); pit != pairs.end(); ++pit)//, ++cit)
    {
        ret.push_back(DBoW2::Result(pit->first, pit->second));// / cit->second));
        wg_item_vec.push_back(pit->first);
    }
        
    // resulting "scores" are now in [-1 best .. 0 worst]	
    
    // sort vector in ascending order of score
    std::sort(ret.begin(), ret.end());
    // (ret is inverted now --the lower the better--)
    std::sort(wg_item_vec.begin(), wg_item_vec.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results)
        ret.resize(max_results);

    // complete and scale score to [0 worst .. 1 best]
    // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) 
	//		for all i | v_i != 0 and w_i != 0 )
	// (Nister, 2006)
    DBoW2::QueryResults::iterator qit;
    for(qit = ret.begin(); qit != ret.end(); qit++) 
    {
        if(qit->Score <= -1.0) // rounding error
        qit->Score = 1.0;
        else
        qit->Score = 1.0 - sqrt(1.0 + qit->Score); // [0..1]
        // the + sign is ok, it is due to - sign in 
        // value = - qvalue * dvalue
    }

    std::sort(ret.begin(), ret.end(), id_compare);
}


void BoWGDatabase::wg_queryCHI_SQUARE(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results, int max_id)
{
    BoWGVector::const_iterator vit;
    typename GroupIFRow::const_iterator rit;
    
    std::map<ItemID, std::pair<double, int> > pairs;
    std::map<ItemID, std::pair<double, int> >::iterator pit;

    std::map<ItemID, std::pair<double, double> > sums; // < sum vi, sum wi >
    std::map<ItemID, std::pair<double, double> >::iterator sit;
    
    for(vit = vec.begin(); vit != vec.end(); ++vit)
    {
        const WordGroupID wg_id = vit->first;
        const WordGroupValue& qvalue = vit->second;
        
        const GroupIFRow& row = m_ifile[wg_id];
        // GroupIFRow are sorted in ascending entry_id order
        
        for(rit = row.begin(); rit != row.end(); ++rit)
        {
        const ItemID entry_id = rit->item_id;
        const WordGroupValue& dvalue = rit->wg_weight;

        if((int)entry_id < max_id || max_id == -1)
        {
            // (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
            // we move the 4 out
            double value = 0;
            if(qvalue + dvalue != 0.0) // words may have weight zero
            value = - qvalue * dvalue / (qvalue + dvalue);
            
            pit = pairs.lower_bound(entry_id);
            sit = sums.lower_bound(entry_id);
            //eit = expected.lower_bound(entry_id);
            if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
            {
                pit->second.first += value;
                pit->second.second += 1;
                //eit->second += dvalue;
                sit->second.first += qvalue;
                sit->second.second += dvalue;
            }
            else
            {
                pairs.insert(pit, 
                    std::map<ItemID, std::pair<double, int> >::value_type(entry_id,
                    std::make_pair(value, 1) ));

                sums.insert(sit, 
                    std::map<ItemID, std::pair<double, double> >::value_type(entry_id,
                    std::make_pair(qvalue, dvalue) ));
            }
        }
        
        } // for each inverted row
    } // for each query word group
        
    // move to vector
    ret.reserve(pairs.size());
    sit = sums.begin();
    for(pit = pairs.begin(); pit != pairs.end(); ++pit, ++sit)
    {
        if(pit->second.second >= DBoW2::MIN_COMMON_WORDS)
        {
        ret.push_back(DBoW2::Result(pit->first, pit->second.first));
        ret.back().nWords = pit->second.second;
        ret.back().sumCommonVi = sit->second.first;
        ret.back().sumCommonWi = sit->second.second;
        ret.back().expectedChiScore = 
            2 * sit->second.second / (1 + sit->second.second);
        }
        wg_item_vec.push_back(pit->first);
    }
        
    // resulting "scores" are now in [-2 best .. 0 worst]	
    // we have to add +2 to the scores to obtain the chi square score
    
    // sort vector in ascending order of score
    std::sort(ret.begin(), ret.end());
    // (ret is inverted now --the lower the better--)
    std::sort(wg_item_vec.begin(), wg_item_vec.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results)
        ret.resize(max_results);

    // complete and scale score to [0 worst .. 1 best]
    DBoW2::QueryResults::iterator qit;
    for(qit = ret.begin(); qit != ret.end(); qit++) 
    {
        // this takes the 4 into account
        qit->Score = - 2. * qit->Score; // [0..1]
        
        qit->chiScore = qit->Score;
    }

    std::sort(ret.begin(), ret.end(), id_compare);
}


void BoWGDatabase::wg_queryKL(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results, int max_id)
{
    BoWGVector::const_iterator vit;
    typename GroupIFRow::const_iterator rit;
    
    std::map<ItemID, double> pairs;
    std::map<ItemID, double>::iterator pit;
    
    for(vit = vec.begin(); vit != vec.end(); ++vit)
    {
        const WordGroupID wg_id = vit->first;
        const WordGroupValue& vi = vit->second;
        
        const GroupIFRow& row = m_ifile[wg_id];
        // GroupIFRow are sorted in ascending entry_id order
        
        for(rit = row.begin(); rit != row.end(); ++rit)
        {
        const ItemID entry_id = rit->item_id;
        const WordGroupValue& wi = rit->wg_weight;

        if((int)entry_id < max_id || max_id == -1)
        {
            double value = 0;
            if(vi != 0 && wi != 0) value = vi * log(vi/wi);
            
            pit = pairs.lower_bound(entry_id);
            if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
            {
            pit->second += value; 
            }
            else
            {
            pairs.insert(pit, 
                std::map<ItemID, double>::value_type(entry_id, value));
            }
        }
        
        } // for each inverted row
    } // for each query word group
        
    // move to vector
    ret.reserve(pairs.size());
    for(pit = pairs.begin(); pit != pairs.end(); ++pit)
    {
        ItemID eid = pit->first;
        double value = 0.0;

        for(vit = vec.begin(); vit != vec.end(); ++vit)
        {
            const WordGroupValue &vi = vit->second;
            const GroupIFRow& row = m_ifile[vit->first];

            if(vi != 0)
            {
                if(row.end() == find(row.begin(), row.end(), eid ))
                {
                    value += vi * (log(vi) - DBoW2::GeneralScoring::LOG_EPS);
                }
            }
        }
        
        pit->second += value;
        
        // to vector
        ret.push_back(DBoW2::Result(pit->first, pit->second));
        wg_item_vec.push_back(pit->first);
    }
        
    // real scores are now in [0 best .. X worst]	
    
    // sort vector in ascending order of score
    std::sort(ret.begin(), ret.end());
    // (ret is inverted now --the lower the better--)
    std::sort(wg_item_vec.begin(), wg_item_vec.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results)
        ret.resize(max_results);

    std::sort(ret.begin(), ret.end(), id_compare);
}


void BoWGDatabase::wg_queryBHATTACHARYYA(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results, int max_id)
{
    BoWGVector::const_iterator vit;
    typename GroupIFRow::const_iterator rit;
    
    std::map<ItemID, std::pair<double, int> > pairs; // <eid, <score, counter> >
    std::map<ItemID, std::pair<double, int> >::iterator pit;
    
    for(vit = vec.begin(); vit != vec.end(); ++vit)
    {
        const WordGroupID wg_id = vit->first;
        const WordGroupValue& qvalue = vit->second;
        
        const GroupIFRow& row = m_ifile[wg_id];
        // GroupIFRow are sorted in ascending entry_id order
        
        for(rit = row.begin(); rit != row.end(); ++rit)
        {
        const ItemID entry_id = rit->item_id;
        const WordGroupValue& dvalue = rit->wg_weight;

        if((int)entry_id < max_id || max_id == -1)
        {
            double value = sqrt(qvalue * dvalue);
        
            pit = pairs.lower_bound(entry_id);
            if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
            {
                pit->second.first += value;
                pit->second.second += 1;
            }
            else
            {
                pairs.insert(pit, 
                    std::map<ItemID, std::pair<double, int> >::value_type(entry_id,
                    std::make_pair(value, 1)));
            }
        }
        
        } // for each inverted row
    } // for each query word group
        
    // move to vector
    ret.reserve(pairs.size());
    for(pit = pairs.begin(); pit != pairs.end(); ++pit)
    {
        if(pit->second.second >= DBoW2::MIN_COMMON_WORDS)
        {
        ret.push_back(DBoW2::Result(pit->first, pit->second.first));
        ret.back().nWords = pit->second.second;
        ret.back().bhatScore = pit->second.first;
        }
        wg_item_vec.push_back(pit->first);
    }
        
    // scores are already in [0..1]	
    
    // sort vector in ascending order of score
    std::sort(ret.begin(), ret.end());
    // (ret is inverted now --the lower the better--)
    std::sort(wg_item_vec.begin(), wg_item_vec.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results)
        ret.resize(max_results);

    std::sort(ret.begin(), ret.end(), id_compare);
}

void BoWGDatabase::wg_queryDOT(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results, int max_id)
{
    BoWGVector::const_iterator vit;
    typename GroupIFRow::const_iterator rit;
    
    std::map<ItemID, double> pairs;
    std::map<ItemID, double>::iterator pit;
    
    for(vit = vec.begin(); vit != vec.end(); ++vit)
    {
        const WordGroupID wg_id = vit->first;
        const WordGroupValue& qvalue = vit->second;
        
        const GroupIFRow& row = m_ifile[wg_id];
        // GroupIFRow are sorted in ascending entry_id order
        
        for(rit = row.begin(); rit != row.end(); ++rit)
        {
        const ItemID entry_id = rit->item_id;
        const WordGroupValue& dvalue = rit->wg_weight;

        if((int)entry_id < max_id || max_id == -1)
        {
            double value = qvalue * dvalue; 
            
            pit = pairs.lower_bound(entry_id);
            if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
            {
                pit->second += value;
            }
            else
            {
                pairs.insert(pit, 
                    std::map<ItemID, double>::value_type(entry_id, value));
            }
        }
        
        } // for each inverted row
    } // for each query word group
        
    // move to vector
    ret.reserve(pairs.size());
    for(pit = pairs.begin(); pit != pairs.end(); ++pit)
    {
        ret.push_back(DBoW2::Result(pit->first, pit->second));
        wg_item_vec.push_back(pit->first);
    }
   
    // scores are the greater the better
    
    // sort vector in ascending order of score
    std::sort(ret.begin(), ret.end());
    // (ret is inverted now --the lower the better--)
    std::sort(wg_item_vec.begin(), wg_item_vec.end());

    // cut vector
    if(max_results > 0 && (int)ret.size() > max_results)
        ret.resize(max_results);

    std::sort(ret.begin(), ret.end(), id_compare);
}

void BoWGDatabase::query_words(DBoW2::QueryResults &in_ret, DBoW2::QueryResults &ret, DBoW2::BowVector &bowVector, int max_results, 
                                    bool use_temporal_score, double prev_weight_th, double temporal_param)
{
    if (m_dtable.size()==0)
        return;

    double norm_term;
    switch(w_scoring_type)
    {
        case DBoW2::L1_NORM:
            norm_term = bowg_scoring.scoreL1(prev_bow_vec, bowVector);
            break;
        case DBoW2::L2_NORM:
            norm_term = bowg_scoring.scoreL2(prev_bow_vec, bowVector);
            break;
        case DBoW2::CHI_SQUARE:
            norm_term = bowg_scoring.scoreChiSquare(prev_bow_vec, bowVector);
            break;
        case DBoW2::KL:
            norm_term = bowg_scoring.scoreKL(prev_bow_vec, bowVector);
            break;
        case DBoW2::BHATTACHARYYA:
            norm_term = bowg_scoring.scoreBhattacharyya(prev_bow_vec, bowVector);
            break;
        case DBoW2::DOT_PRODUCT:
            norm_term = bowg_scoring.scoreDot(prev_bow_vec, bowVector);
            break;
    }
    if (norm_term < min_prev_score)
        norm_term = 10;

    for (size_t i = 0; i<in_ret.size(); i++)
    {
        BoWG::ItemID img_id = in_ret[i].Id;
        DBoW2::Result co_result;
        co_result.Id = img_id;
        double cur_score = in_ret[i].Score / norm_term;

        if (!use_temporal_score)
            co_result.Score = cur_score;
        else {
            auto it = last_query_res.find(img_id - 1);
            if (it != last_query_res.end()) {
                double prev_weight = prev_weight_th / ((it->second - cur_score)*(it->second - cur_score) / (temporal_param*temporal_param) + 1);
                co_result.Score = prev_weight*it->second + (1.0-prev_weight)*cur_score;
            }
            else {
                co_result.Score = cur_score;
            }
        }
        ret.push_back(co_result);
    }

    // in VINS, don't need sequential output since we don't use the island matching
    std::sort(ret.begin(), ret.end(), id_compare);
    store_last_res(ret);

    std::sort(ret.begin(), ret.end(), compare);
    if (ret.size() > static_cast<size_t>(max_results))
        ret.resize(max_results);


    return;
}


void BoWGDatabase::query_bowg(DBoW2::QueryResults &in_ret, DBoW2::QueryResults &ret, DBoW2::BowVector &bowVector, BoWGVector &bowgVector, 
                        int max_results, int max_id, double w_weight, bool use_temporal_score, double prev_weight_th, double temporal_param)
{
    if (m_dtable.size()==0)
        return;

    DBoW2::QueryResults wg_score_vec;

    std::sort(in_ret.begin(), in_ret.end(), id_compare);

    double norm_term;
    switch(w_scoring_type)
    {
        case DBoW2::L1_NORM:
            norm_term = bowg_scoring.scoreL1(prev_bow_vec, bowVector);
            break;
        case DBoW2::L2_NORM:
            norm_term = bowg_scoring.scoreL2(prev_bow_vec, bowVector);
            break;
        case DBoW2::CHI_SQUARE:
            norm_term = bowg_scoring.scoreChiSquare(prev_bow_vec, bowVector);
            break;
        case DBoW2::KL:
            norm_term = bowg_scoring.scoreKL(prev_bow_vec, bowVector);
            break;
        case DBoW2::BHATTACHARYYA:
            norm_term = bowg_scoring.scoreBhattacharyya(prev_bow_vec, bowVector);
            break;
        case DBoW2::DOT_PRODUCT:
            norm_term = bowg_scoring.scoreDot(prev_bow_vec, bowVector);
            break;
    }
    if (norm_term < min_prev_score)
        norm_term = 10;


    double bowg_norm_term;
    DBoW2::QueryResults wg_ret;
    std::vector<ItemID> wg_item_vec;
    switch(wg_scoring_type)
    {
        case DBoW2::L1_NORM:
            bowg_norm_term = bowg_scoring.scoreL1(prev_bowg_vec, bowgVector);
            wg_queryL1(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::L2_NORM:
            bowg_norm_term = bowg_scoring.scoreL2(prev_bowg_vec, bowgVector);
            wg_queryL2(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::CHI_SQUARE:
            bowg_norm_term = bowg_scoring.scoreChiSquare(prev_bowg_vec, bowgVector);
            wg_queryCHI_SQUARE(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::KL:
            bowg_norm_term = bowg_scoring.scoreKL(prev_bowg_vec, bowgVector);
            wg_queryKL(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::BHATTACHARYYA:
            bowg_norm_term = bowg_scoring.scoreBhattacharyya(prev_bowg_vec, bowgVector);
            wg_queryBHATTACHARYYA(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::DOT_PRODUCT:
            bowg_norm_term = bowg_scoring.scoreDot(prev_bowg_vec, bowgVector);
            wg_queryDOT(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
    }
    if (bowg_norm_term < min_prev_wg_score)
        bowg_norm_term = 10;

    int wg_idx = 0;
    for (size_t i = 0; i<in_ret.size(); i++)
    {
        BoWG::ItemID img_id = in_ret[i].Id;
        DBoW2::Result co_result;
        co_result.Id = img_id;
        double norm_w_score = in_ret[i].Score / norm_term;

        double wg_score;
        if (std::find(wg_item_vec.begin(), wg_item_vec.end(), img_id) != wg_item_vec.end()) {
            wg_score = wg_ret[wg_idx].Score;
            wg_idx++;
        }
        else {
            wg_score = 0.0;
        }
        double norm_wg_score = wg_score / bowg_norm_term;


        DBoW2::Result wg_result;
        wg_result.Id = img_id;
        wg_result.Score = wg_score;
        wg_score_vec.push_back(wg_result);

        double combined_score = w_weight*norm_w_score + (1.0-w_weight)*norm_wg_score;

        if (!use_temporal_score)
            co_result.Score = combined_score;
        else {
            auto it = last_query_res.find(img_id - 1);
            if (it != last_query_res.end()) {
                double prev_weight = prev_weight_th / ((it->second - combined_score)*(it->second - combined_score) / (temporal_param*temporal_param) + 1);
                co_result.Score = prev_weight*it->second + (1.0-prev_weight)*combined_score;
            }
            else {
                co_result.Score = combined_score;
            }
        }
        ret.push_back(co_result);
    }

    // in VINS, don't need sequential output since we don't use the island matching
    std::sort(ret.begin(), ret.end(), id_compare);
    store_last_res(ret);

    std::sort(ret.begin(), ret.end(), compare);
    if (ret.size() > static_cast<size_t>(max_results))
        ret.resize(max_results);
    
    return;
}

void BoWGDatabase::query_bowg(DBoW2::QueryResults &in_ret, DBoW2::QueryResults &ret, DBoW2::BowVector &bowVector, BoWGVector &bowgVector, std::vector<int> dist_vec, 
            int max_results, int max_id, double w_weight, double wg_weight, bool use_temporal_score, double prev_weight_th, double temporal_param)
{
    if (m_dtable.size()==0)
        return;

    DBoW2::QueryResults wg_score_vec;
    DBoW2::QueryResults dist_score_vec;

    std::sort(in_ret.begin(), in_ret.end(), id_compare);

    double norm_term;
    switch(w_scoring_type)
    {
        case DBoW2::L1_NORM:
            norm_term = bowg_scoring.scoreL1(prev_bow_vec, bowVector);
            break;
        case DBoW2::L2_NORM:
            norm_term = bowg_scoring.scoreL2(prev_bow_vec, bowVector);
            break;
        case DBoW2::CHI_SQUARE:
            norm_term = bowg_scoring.scoreChiSquare(prev_bow_vec, bowVector);
            break;
        case DBoW2::KL:
            norm_term = bowg_scoring.scoreKL(prev_bow_vec, bowVector);
            break;
        case DBoW2::BHATTACHARYYA:
            norm_term = bowg_scoring.scoreBhattacharyya(prev_bow_vec, bowVector);
            break;
        case DBoW2::DOT_PRODUCT:
            norm_term = bowg_scoring.scoreDot(prev_bow_vec, bowVector);
            break;
    }
    if (norm_term < min_prev_score)
        norm_term = 10;


    double dist_norm_term = bowg_scoring.dis_score(prev_dist_vec, dist_vec);
    if (dist_norm_term < min_prev_dist_score)
        dist_norm_term = 10;

    double bowg_norm_term;
    DBoW2::QueryResults wg_ret;
    std::vector<ItemID> wg_item_vec;
    switch(wg_scoring_type)
    {
        case DBoW2::L1_NORM:
            bowg_norm_term = bowg_scoring.scoreL1(prev_bowg_vec, bowgVector);
            wg_queryL1(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::L2_NORM:
            bowg_norm_term = bowg_scoring.scoreL2(prev_bowg_vec, bowgVector);
            wg_queryL2(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::CHI_SQUARE:
            bowg_norm_term = bowg_scoring.scoreChiSquare(prev_bowg_vec, bowgVector);
            wg_queryCHI_SQUARE(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::KL:
            bowg_norm_term = bowg_scoring.scoreKL(prev_bowg_vec, bowgVector);
            wg_queryKL(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::BHATTACHARYYA:
            bowg_norm_term = bowg_scoring.scoreBhattacharyya(prev_bowg_vec, bowgVector);
            wg_queryBHATTACHARYYA(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
        case DBoW2::DOT_PRODUCT:
            bowg_norm_term = bowg_scoring.scoreDot(prev_bowg_vec, bowgVector);
            wg_queryDOT(bowgVector, wg_ret, wg_item_vec, max_results, max_id);
            break;
    }
    if (bowg_norm_term < min_prev_wg_score)
        bowg_norm_term = 10;

    int wg_idx = 0;
    for (size_t i = 0; i<in_ret.size(); i++)
    {
        BoWG::ItemID img_id = in_ret[i].Id;
        DBoW2::Result co_result;
        co_result.Id = img_id;
        double norm_w_score = in_ret[i].Score / norm_term;

        double wg_score;
        if (std::find(wg_item_vec.begin(), wg_item_vec.end(), img_id) != wg_item_vec.end()) {
            wg_score = wg_ret[wg_idx].Score;
            wg_idx++;
        }
        else {
            wg_score = 0.0;
        }
        double norm_wg_score = wg_score / bowg_norm_term;

        std::vector<int> tmp_distVector = dis_table[img_id];
        double dist_score = bowg_scoring.dis_score(dist_vec, tmp_distVector);
        double norm_dist_score = dist_score / dist_norm_term;

        DBoW2::Result wg_result;
        wg_result.Id = img_id;
        wg_result.Score = wg_score;
        wg_score_vec.push_back(wg_result);

        DBoW2::Result dist_result;
        dist_result.Id = img_id;
        dist_result.Score = dist_score;
        dist_score_vec.push_back(dist_result);

        double combined_score = w_weight*norm_w_score + wg_weight*norm_wg_score + (1.0-w_weight-wg_weight)*norm_dist_score;

        if (!use_temporal_score)
            co_result.Score = combined_score;
        else {
            auto it = last_query_res.find(img_id - 1);
            if (it != last_query_res.end()) {
                double prev_weight = prev_weight_th / ((it->second - combined_score)*(it->second - combined_score) / (temporal_param*temporal_param) + 1);
                co_result.Score = prev_weight*it->second + (1.0-prev_weight)*combined_score;
            }
            else {
                co_result.Score = combined_score;
            }
        }
        ret.push_back(co_result);
    }

    // in VINS, don't need sequential output since we don't use the island matching
    std::sort(ret.begin(), ret.end(), id_compare);
    store_last_res(ret);

    std::sort(ret.begin(), ret.end(), compare);
    if (ret.size() > static_cast<size_t>(max_results))
        ret.resize(max_results);
    

    return;
}

bool BoWGDatabase::matchGrouping(DBoW2::QueryResults &ret, std::vector<std::vector<int>> &groupMatches, double candidates_th, int max_intraisland_gap)
{
    int island_idx = 0;
    int prev_item_idx;
    bool first_candidate = true;
    for (size_t i = 0; i < ret.size(); i++)
    {
        // consider as a loop closure candidate
        if (ret[i].Score > candidates_th) {
            ItemID loop_ID = ret[i].Id;
            if (first_candidate) {
                groupMatches.emplace_back();
                groupMatches[island_idx].push_back(i);
                prev_item_idx = loop_ID;
                first_candidate = false;
                continue;
            }
            // smaller than the time gap threshold, not a new island
            if (loop_ID - prev_item_idx < max_intraisland_gap) {
                groupMatches[island_idx].push_back(i);
                prev_item_idx = loop_ID;
            }
            else { // new island
                groupMatches.emplace_back();
                island_idx = island_idx + 1;
                groupMatches[island_idx].push_back(i);
                prev_item_idx = loop_ID;    
            }
        }
    }
    
    if (groupMatches.empty())
        return false;

    return true;
}


int BoWGDatabase::islandMatching(DBoW2::QueryResults &ret, std::vector<std::vector<int>> &groupMatches)
{
    double max_score = 0;
    int max_idx = 0;
    for (size_t i = 0; i < groupMatches.size(); i++)
    {
        double island_score = 0;
        for (size_t j = 0; j < groupMatches[i].size(); j++)
        {  
            int idx = groupMatches[i][j];
            island_score = island_score + ret[idx].Score;
        }
        if (island_score > max_score)
        {
            max_score = island_score;
            max_idx = i;
        }
    }
    Islands_map[cur_image_id] = groupMatches[max_idx];
    return max_idx;
}


bool BoWGDatabase::temporalConsistency(DBoW2::QueryResults &ret, std::vector<int> &island, int k, int max_group_gap, int max_query_gap){
    // temporal consistency check, consider previous k islands 
    ItemID island_start = ret[island[0]].Id;
    ItemID island_end = ret[island.back()].Id;

    int last_query_id = cur_image_id - 1;

    for (int i = 0; i < k; i++)
    {
        std::vector<int> prev_island;
        bool temp_flag = false;
        DBoW2::QueryResults last_res;
        for (int j = 0; j < max_query_gap; j++)
        {
            auto it = Islands_map.find(last_query_id - j);
            if (it != Islands_map.end()) {
                prev_island = it->second;
                last_res = res_table[last_query_id - j];
                temp_flag = true;
                last_query_id = last_query_id - j - 1;
                break;
            }
        }

        if (!temp_flag) {
            std::cout << "temporal consistency failed" << std::endl;
            return false;
        }

        ItemID prev_start = last_res[prev_island[0]].Id;
        ItemID prev_end = last_res[prev_island.back()].Id;

        if ((island_start <= prev_start && prev_start <= island_end) || (prev_start <= island_start && island_start <= prev_end))
        {
            island_start = prev_start;
            island_end = prev_end;
        }
        else {
            int d1 = (int)prev_start - (int)island_end;
            int d2 = (int)island_start - (int)prev_end;
            int gap = (d1 > d2 ? d1 : d2);
            if (gap > max_group_gap)
            {
                std::cout << "temporal consistency failed" << std::endl;
                return false;
            }
            island_start = prev_start;
            island_end = prev_end;
        }
    }

    return true;
}


int BoWGDatabase::get_num_wg(void)
{
    return voc_wg.size();
}


} //namespace BOWG