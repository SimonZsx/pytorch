6.19
Set commit to 174e1ba3b8ed640c09b5ff927227e765193d0cad, PyTroch release 1.5.0

devdocker pytorch:1.5.0-dev 


torch CMakelists 245

Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  std::cout<<"addmm hello"<<std::endl;
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<AddmmBackward>(new AddmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
    grad_fn->mat1_ = SavedVariable(mat1, false);

    grad_fn->saved_variables.push_back(& grad_fn->mat1_);

    grad_fn->mat2_ = SavedVariable(mat2, false);

    grad_fn->saved_variables.push_back(& grad_fn->mat2_);


    grad_fn->alpha = alpha;
    grad_fn->mat2_sizes = mat2.sizes().vec();
    grad_fn->beta = beta;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::addmm(self_, mat1_, mat2_, beta, alpha);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }

  std::cout<<tensordb.get_size()<<std::endl;
  tensordb.insert(uni_id,grad_fn);
  //tensordb.insert_tensor(grad_fn, grad_fn->mat1_);
  //tensordb.insert_tensor(grad_fn, grad_fn->mat2_);

  
  uni_id++;
  tensordb.print();
  
  //grad_fn->mat1_.set_data(grad_fn->mat1_.get_data().cpu());
  grad_fn->mat2_.non_block_cpu();
  //std::cout<<"swap mat2_ to cpu()"<<std::endl;
  //std::cout<<"creation mat1_ cuda: "<<grad_fn->mat1_.get_data().is_cuda()<<std::endl;
  std::cout<<"storage size :"<<grad_fn->mat1_.get_data().storage().nbytes()<<std::endl;
  //std::cout<<"change grad_fn"<<std::endl;
  //auto r = grad_fn->mat1_.get_data().cpu();

  return result;
}


variable_list AddmmBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::cout<<"AddmmBackward"<<std::endl;
  std::cout<<"cuda: "<<mat1_.get_data().is_cuda()<<std::endl;
  std::cout<<"cuda: "<<mat2_.get_data().is_cuda()<<std::endl;

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat1_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto mat1 = mat1_.unpack();
  auto mat2 = mat2_.unpack();
  std::cout<<"grad cuda: "<<grad.is_cuda()<<std::endl;
  std::cout<<"mat1 cuda: "<<mat1.is_cuda()<<std::endl;
  std::cout<<"mat2 cuda: "<<mat2.is_cuda()<<std::endl;

  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mat1_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat1_backward(grad, mat2, mat1, alpha)) : Tensor();
    copy_range(grad_inputs, mat1_ix, grad_result);
    std::cout<<"shoud compute 1"<<std::endl;
  }
  if (should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat2_backward(grad, mat1, mat2_sizes, mat2.strides(), alpha)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
    std::cout<<"shoud compute 2"<<std::endl;

  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
    std::cout<<"shoud compute 3"<<std::endl;

  }
  return grad_inputs;
}


struct TORCH_API AddmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmmBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout<<"release varaibles"<<std::endl;

    std::cout<<"release mat1_ cuda: "<<mat1_.get_data().is_cuda()<<std::endl;
    std::cout<<"release mat2_ cuda: "<<mat2_.get_data().is_cuda()<<std::endl;

    mat1_.reset_data();
    mat1_.reset_grad_function();
    mat2_.reset_data();
    mat2_.reset_grad_function();
  }

  SavedVariable mat1_;
  SavedVariable mat2_;
  Scalar alpha;
  std::vector<int64_t> mat2_sizes;
  Scalar beta;

  std::vector<void *> saved_variables;

};