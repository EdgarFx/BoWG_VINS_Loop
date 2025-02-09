/**
 * File: BoWGVector.h
 * Date: August 2024
 * Author: Xiang Fei
 * Description: bag of word groups vector, used to describe images
*/

#ifndef __BOWG_VECTOR__
#define __BOWG_VECTOR__

#include <iostream>
#include <map>
#include <vector>

namespace BoWG {
    
// ID of Word Groups
typedef unsigned int WordGroupID;

// Value of a Word Group
typedef double WordGroupValue;

// Vector of word groups to represent images
class BoWGVector:
    public std::map<WordGroupID, WordGroupValue>
{
public:

    /**
    * Constructor of BoWGVector
    */
    BoWGVector(void);

    /**
    * Destructor of BoWGVector
    */
    ~BoWGVector(void);

    /**
    * Add a value to a word groups value existing in the vector, or create a new 
    * word group with the given value, used for TF-IDF.
    * @param id word group id to look for
    * @param v value to create the word group with, or to add to existing word group
    */
    void addWeight(WordGroupID id, WordGroupValue v);

    /**
     * Adds a word group with a value to the vector only if this does not exist yet
     * used for IDF or BINARY
     * @param id word group id to look for
     * @param v value to give to the word group if this does not exist
    */
    void addIfNotExist(WordGroupID id, WordGroupValue v);

    /**
     * L-Normalizes the values in the vector
    */
    void normalize();

    /**
     * Prints the content of the bowg vector
     * @param out stream'
     * @param v
    */
    friend std::ostream& operator<<(std::ostream &out, const BoWGVector &v);
};

} // namespace BoWG

#endif