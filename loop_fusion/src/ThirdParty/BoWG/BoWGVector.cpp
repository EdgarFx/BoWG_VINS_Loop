/**
 * File: BoWGVector.cpp
 * Date: July 2023
 * Author: August 2024
 * Description: bag of word groups vector, used to describe images
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "BoWGVector.h"

namespace BoWG {

// --------------------------------------------------------------------------

BoWGVector::BoWGVector(void)
{
}

// --------------------------------------------------------------------------

BoWGVector::~BoWGVector(void)
{
}

// --------------------------------------------------------------------------

void BoWGVector::addWeight(WordGroupID id, WordGroupValue v)
{
    BoWGVector::iterator vit = this->lower_bound(id);

    if(vit != this->end() && !(this->key_comp()(id, vit->first)))
    {
        vit->second += v;
    }
    else
    {
        this->insert(vit, BoWGVector::value_type(id, v));
    }
}

// --------------------------------------------------------------------------

void BoWGVector::addIfNotExist(WordGroupID id, WordGroupValue v)
{
    BoWGVector::iterator vit = this->lower_bound(id);

    if(vit == this->end() || (this->key_comp()(id, vit->first)))
    {
        this->insert(vit, BoWGVector::value_type(id, v));
    }
}

// --------------------------------------------------------------------------

void BoWGVector::normalize()
{
    double norm = 0.0;
    BoWGVector::iterator it;

    for(it = begin(); it != end(); ++it)
        norm += it->second * it->second;
    norm = sqrt(norm);

    if(norm > 0.0)
    {
        for(it = begin(); it != end(); ++it)
            it->second /= norm;
    }
}

// --------------------------------------------------------------------------

std::ostream& operator<< (std::ostream &out, const BoWGVector &v)
{
  BoWGVector::const_iterator vit;
  std::vector<unsigned int>::const_iterator iit;
  unsigned int i = 0; 
  const unsigned int N = v.size();
  for(vit = v.begin(); vit != v.end(); ++vit, ++i)
  {
    out << "<" << vit->first << ", " << vit->second << ">";
    
    if(i < N-1) out << ", ";
  }
  return out;
}

} // namespace BoWG