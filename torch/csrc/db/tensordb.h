#ifndef TENSOR_DB_H
#define TENSOR_DB_H

#include <iostream> 
#include <iterator> 
#include <map> 

#include <ATen/TypeDefault.h>
#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

#include "torch/csrc/autograd/saved_variable.h"


using namespace at;
using namespace torch::autograd;
//using namespace torch::autograd::generated;


namespace torch { namespace tensordb{

class TensorDatabase{


    private:
        std::size_t dbsize;
        std::vector<int> data;

        std::map<int, std::shared_ptr<Node>> tensormap; 
        //std::map<std::shared_ptr<Node>, std::vector<SavedVariable>> activations;

    public:
        TensorDatabase(){

            

        }
        TensorDatabase(std::size_t R): dbsize(R){

        }
        int operator()(size_t r, size_t c) const { // member function definition
            return data[r*dbsize+c];
        }
        int& operator()(size_t r, size_t c) {  // another member function definition
            return data[r*dbsize+c];
        }

        void insert(int i, std::shared_ptr<Node> ptr){
                        //std::cout << "\nTensor map : \n"; 
            
            
            if (i>=2){
            for (auto & it: (tensormap.find(i-1)->second.get())->saved_variables){
                std::cout << "swap to cpu : "<<tensormap.find(i-1)->second.get()->name()<<std::endl; 

                ((SavedVariable *)it)->cpu();

                std::cout << "is cuda:" << ((SavedVariable *)it)->get_data().is_cuda()<<std::endl;
            }}
            tensormap.insert(std::pair<int, std::shared_ptr<Node>>(i, ptr));
            print();

        }

        // void insert_tensor(std::shared_ptr<Node> ptr, std::shared_ptr<SavedVariable> sv){
        //     if (activations.find(ptr)!=activations.end()){
        //         activations[ptr].push_back(sv);
        //     }
        //     else{
        //         std::vector<SavedVariable> temp;
        //         temp.push_back(sv);
        //         activations.insert(std::pair<std::shared_ptr<Node>,std::vector<std::shared_ptr<SavedVariable>>>(ptr, temp));
        //     }          
        // }

        void print(){
            std::map<int, std::shared_ptr<Node>>::iterator itr; 
            std::cout << "\nTensor Database : \n"; 

            for (itr = tensormap.begin(); itr != tensormap.end(); ++itr) { 
                std::cout << '\t' << itr->first << '\t' << itr->second.get()->name() << '\n'; 
            } 
        }
        
        std::size_t get_size(){
            return dbsize;
        }

};

}}//end namespace




#endif