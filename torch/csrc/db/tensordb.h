#ifndef TENSOR_DB_H
#define TENSOR_DB_H

#include <iostream> 
#include <iterator> 
#include <map> 
#include <sstream> 
#include <string>

#include <chrono>

#include <future>
#include <mutex>


#include <ATen/TypeDefault.h>
#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

#include "torch/csrc/autograd/saved_variable.h"

using namespace at;
using namespace torch::autograd;
using namespace std;
//using namespace torch::autograd::generated;


namespace torch { namespace tensordb{

class TensorDatabase{


    private:
        std::size_t dbsize;
        std::vector<int> data;
        int missed;

        std::map<int, std::shared_ptr<Node>> tensormap; 
        //std::map<std::shared_ptr<Node>, std::vector<SavedVariable>> activations;


       // std::map vector<std::shared_ptr<Node>>;
        std::vector<std::future<void>> pending_futures;

        //indicate whether a tensor should be swapped out.
        std::vector<int> flags;

        map<string, int> swapping_layers; 


        int pipeline_id;
        bool is_forward;


    public:
        TensorDatabase(){
            missed = 0;

        
        }
        TensorDatabase(std::size_t R): dbsize(R){

        }

        void set_pipeline(int p_id, bool is_f){
            pipeline_id = p_id;
            is_forward = is_f;

        }

        void insert(int i, std::shared_ptr<Node> ptr){
                        //std::cout << "\nTensor map : \n"; 
            // if( ptr.get()->saved_variables.size()==0){
            //     return;
            // }
            // if (ptr.get()->name()== "SelectBackward"){
            //     return;
            // }

            std::chrono::steady_clock sc;  
            auto start = sc.now(); 

            if (i>=2){
            for (auto & it: (tensormap.find(i-1)->second.get())->saved_variables){
                if (tensormap.find(i-1)->second.get()->name() == "AddmmBackward") {break;}
                if (tensormap.find(i-1)->second.get()->name() == "MaxPool2DWithIndicesBackward") {break;}
                if (tensormap.find(i-1)->second.get()->name() == "MaxPool2DWithIndicesBackward") {break;}
                std::cout << "swap to cpu : "<<tensormap.find(i-1)->second.get()->name()<<std::endl; 
                //Sync 
                //((SavedVariable *)it)->non_block_cpu();
                //Async
                auto f =std::async(std::launch::async, &SavedVariable::non_block_cpu,(SavedVariable *)it);

                pending_futures.push_back(std::move(f));

                //std::cout << "is cuda:" << ((SavedVariable *)it)->get_data().is_cuda()<<std::endl;
            }}
            tensormap.insert(std::pair<int, std::shared_ptr<Node>>(i, ptr));
            print();

            auto end = sc.now();     
            auto time_span = static_cast<std::chrono::duration<double>>(end - start);   
            std::cout<<"Operation took: "<<time_span.count()<<" seconds \n";
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


        std::string get_tensormap(){

            std::stringstream ss;

            std::map<int, std::shared_ptr<Node>>::iterator itr; 


            for (itr = tensormap.begin(); itr != tensormap.end(); ++itr) { 
                ss  << itr->first << itr->second.get()->name() << '\n'; 
            }

            return ss.str();
        }
        
        std::size_t get_size(){
            return dbsize;
        }

};

}}//end namespace




#endif