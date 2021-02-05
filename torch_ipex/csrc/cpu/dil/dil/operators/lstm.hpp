#ifndef DIL_OPERATORS_LSTM_HPP
#define DIL_OPERATORS_LSTM_HPP

namespace dil {

struct lstm_forward : public dnnl::lstm_forward, utils::computation_cache<dnnl::lstm_forward::primitive_desc> {

  using super = dnnl::lstm_forward;

  static void compute(const dims& output_sizes,
                      const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& src_iter_c,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      tensor& dst_layer,
                      tensor& dst_iter,
                      tensor& dst_iter_c,
                      const bool reverse = false,
                      prop_kind aprop = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    auto prep_start = std::chrono::high_resolution_clock::now();

    bool with_workspace = aprop == prop_kind::forward_training;
    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc().to_type(src_layer.get_data_type());
    auto src_iter_c_desc = src_iter_c.get_desc();

    auto weights_layer_desc = weights_layer.get_desc();
    auto weights_iter_desc = weights_iter.get_desc();

    // If the weight is prepacked, the weight will be padded(fp32 & bf16) or blocked(int8), which is not dense
    // If not prepacked:     
    //  use any format for weights
    //  For accuracy consideration, weight remains fp32 when doing training,
    //  so it is necessary to align weights data type with src in here.
    if (weights_layer_desc.is_dense()) {
      weights_layer_desc = weights_layer_desc.to_format_any().to_type(src_layer.get_data_type());
    }
    if (weights_iter_desc.is_dense()) {
      weights_iter_desc = weights_iter_desc.to_format_any().to_type(src_layer.get_data_type());
    }

    auto bias_desc = bias.get_desc();
    tensor::desc dst_layer_desc(output_sizes, src_layer.get_data_type(), tag::tnc);

    auto primitive_start = std::chrono::high_resolution_clock::now();

    auto pd = get_primitive_desc(
      src_layer_desc, src_iter_desc, src_iter_c_desc, 
      weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, 
      reverse, aprop, aengine);

    auto primitive_end = std::chrono::high_resolution_clock::now();


    std::cout << "reorder start\n";
    auto expected_src_iter = src_iter.reorder_if_differ_in(pd.src_iter_desc());


    auto reorder_end = std::chrono::high_resolution_clock::now();

    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());


    auto wreorder_end = std::chrono::high_resolution_clock::now();
    std::cout << "reorder end\n";

    dst_layer.reinit_if_possible(pd.dst_layer_desc());
    dst_iter.reinit_if_possible(pd.dst_iter_desc());
    dst_iter_c.reinit_if_possible(pd.dst_iter_c_desc());

    auto reinit_end = std::chrono::high_resolution_clock::now();

    exec_args args {{DNNL_ARG_SRC_LAYER, src_layer},
                    {DNNL_ARG_SRC_ITER, expected_src_iter},
                    {DNNL_ARG_SRC_ITER_C, src_iter_c},
                    {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                    {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                    {DNNL_ARG_BIAS, bias},
                    {DNNL_ARG_DST_LAYER, dst_layer},
                    {DNNL_ARG_DST_ITER, dst_iter},
                    {DNNL_ARG_DST_ITER_C, dst_iter_c}};

    if (with_workspace) {
      dst_layer.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst_layer.get_workspace()});
    }

    auto prep_end = std::chrono::high_resolution_clock::now();


    super(pd).execute(stream::default_stream(), args);
  
    auto execute_end = std::chrono::high_resolution_clock::now();

    auto dur_prep = prep_end - prep_start;
    auto dur_prep1 = primitive_start - prep_start;
    auto dur_primitive = primitive_end - primitive_start;
    auto dur_prep2 = prep_end - reinit_end;
    auto dur_reorder = reorder_end - primitive_end;
    auto dur_wreorder = wreorder_end - reorder_end;
    auto dur_reinit = reinit_end - wreorder_end;
    auto dur_exec = execute_end - prep_end;

    std::cout << "Lstm prep time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(dur_prep).count()/ 1000000.0 << " ms, "
              << "\n"
              << "dur prep1% = " << dur_prep1.count()*100.0/dur_prep.count() << "%,"
              << "dur prep1 = " << std::chrono::duration_cast<std::chrono::nanoseconds>(dur_prep1).count() / 1000000.0 << "ms, "
              << "\n"
              
              << "dur primitive% = " << dur_primitive.count()*100.0/dur_prep.count() << "%,"
              << "dur primitive = " <<  std::chrono::duration_cast<std::chrono::nanoseconds>(dur_primitive).count() / 1000000.0 << "ms,"
              << "\n"
              
              << "dur reorder% = " << dur_reorder.count()*100.0/dur_prep.count() << "%,"
              << "dur reorder = " << std::chrono::duration_cast<std::chrono::nanoseconds>(dur_reorder).count() / 1000000.0 << "ms,"
              << "\n"

              << "dur wreorder% = " << dur_wreorder.count()*100.0/dur_prep.count() << "%,"
              << "dur wreorder = " << std::chrono::duration_cast<std::chrono::nanoseconds>(dur_wreorder).count() / 1000000.0 << "ms,"
              << "\n"

              << "dur reinit = " << dur_reinit.count()*100.0/dur_prep.count() << "%,"
              << "dur reinit = " << std::chrono::duration_cast<std::chrono::nanoseconds>(dur_reinit).count() / 1000000.0 << "ms,"
              << "\n"



              << "dur prep2 = " << dur_prep2.count()*100.0/dur_prep.count() << "%,"
              << "dur prep2 = " << std::chrono::duration_cast<std::chrono::nanoseconds>(dur_prep2).count() / 1000000.0 << "ms,"

              << "\n"


              << " exec time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(dur_exec).count()/ 1000000.0 << " ms." << std::endl;
  
  
  }
  
  static std::tuple<tensor::desc, tensor::desc> expected_weights_desc(const dims& output_sizes,
                      const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& src_iter_c,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      const bool reverse = false,
                      prop_kind aprop = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {

    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc().to_type(src_layer.get_data_type());
    auto src_iter_c_desc = src_iter_c.get_desc();

    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    
    auto bias_desc = bias.get_desc();
    tensor::desc dst_layer_desc(output_sizes, src_layer.get_data_type(), tag::tnc);

    auto pd = get_primitive_desc(
      src_layer_desc, src_iter_desc, src_iter_c_desc, 
      weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, 
      reverse, aprop, aengine);

    auto expected_weights_layer = pd.weights_layer_desc();
    auto expected_weights_iter = pd.weights_iter_desc();

    return std::make_tuple(expected_weights_layer, expected_weights_iter);
  }

  static primitive_desc get_primitive_desc(
    const tensor::desc& src_layer_desc,
    const tensor::desc& src_iter_desc,
    const tensor::desc& src_iter_c_desc,
    const tensor::desc& weights_layer_desc,
    const tensor::desc& weights_iter_desc,
    const tensor::desc& bias_desc,
    const tensor::desc& dst_layer_desc,
    const bool reverse = false,
    
    prop_kind aprop = prop_kind::forward,

    const engine& aengine = engine::cpu_engine()) {
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    // return primitive_desc(
    //     {aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
    //      weights_layer_desc, weights_iter_desc, bias_desc,
    //      dst_layer_desc, src_iter_desc, src_iter_c_desc},
    //      aengine);


    auto key = utils::create_key(aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
                                 weights_layer_desc, weights_iter_desc, bias_desc,
                                 dst_layer_desc);
    return fetch_or_create(key, [&]() {
      return primitive_desc({aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, src_iter_desc, src_iter_c_desc},
         aengine);
    
    });

  }  
};

struct lstm_backward : public dnnl::lstm_backward {

  using super = dnnl::lstm_backward;

  static void compute(const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& src_iter_c,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      const tensor& dst_layer,
                      const tensor& dst_iter,
                      const tensor& dst_iter_c,
                      const tensor& diff_dst_layer,
                      const tensor& diff_dst_iter,
                      const tensor& diff_dst_iter_c,
                      tensor& diff_src_layer,
                      tensor& diff_src_iter,
                      tensor& diff_src_iter_c,
                      tensor& diff_weights_layer,
                      tensor& diff_weights_iter,
                      tensor& diff_bias,
                      const bool reverse = false,
                      const engine& aengine = engine::cpu_engine()) {
    auto aprop = prop_kind::backward;
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;
    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc().to_type(src_layer.get_data_type());
    auto src_iter_c_desc = src_iter_c.get_desc();
    // use any format for weights
    // align weights data type with src
    auto weights_layer_desc = weights_layer.get_desc().to_format_any().to_type(src_layer.get_data_type());
    auto weights_iter_desc = weights_iter.get_desc().to_format_any().to_type(src_layer.get_data_type());
    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();
    auto dst_iter_c_desc = dst_iter_c.get_desc();

    auto diff_src_layer_desc = src_layer_desc.to_type(data_type::f32);
    auto diff_src_iter_desc = src_iter_desc.to_type(data_type::f32);
    auto diff_src_iter_c_desc = src_iter_c_desc.to_type(data_type::f32);
    auto diff_weights_layer_desc = weights_layer_desc.to_type(data_type::f32);
    auto diff_weights_iter_desc = weights_iter_desc.to_type(data_type::f32);
    auto diff_bias_desc = bias_desc.to_type(data_type::f32);
    auto diff_dst_layer_desc = dst_layer_desc.to_type(data_type::f32);
    auto diff_dst_iter_desc = dst_iter_desc.to_type(data_type::f32);
    auto diff_dst_iter_c_desc = dst_iter_c_desc.to_type(data_type::f32);

    auto forward_hints =
        dnnl::lstm_forward::primitive_desc(
            {prop_kind::forward_training, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc, dst_iter_c_desc},
        aengine);

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc, dst_iter_c_desc,
         diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc,
         diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc,
         diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc},
        aengine, forward_hints);

    auto expected_src_iter = src_iter.reorder_if_differ_in(pd.src_iter_desc());
    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    diff_src_layer.reinit_if_possible(pd.diff_src_layer_desc());
    diff_src_iter.reinit_if_possible(pd.diff_src_iter_desc());
    diff_src_iter_c.reinit_if_possible(pd.diff_src_iter_c_desc());
    //workaround: diff_weights_layer, diff_weights_iter and diff_bias need to clear before operation begin.
    diff_weights_layer.zero_init(pd.diff_weights_layer_desc());
    diff_weights_iter.zero_init(pd.diff_weights_iter_desc());
    diff_bias.zero_init(pd.diff_bias_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, expected_src_iter},
                       {DNNL_ARG_SRC_ITER_C, src_iter_c},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter},
                       {DNNL_ARG_DST_ITER_C, dst_iter_c},
                       {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer},
                       {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter},
                       {DNNL_ARG_DIFF_SRC_ITER_C, diff_src_iter_c},
                       {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer},
                       {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter},
                       {DNNL_ARG_DIFF_BIAS, diff_bias},
                       {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer},
                       {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter},
                       {DNNL_ARG_DIFF_DST_ITER_C, diff_dst_iter_c},
                       {DNNL_ARG_WORKSPACE, dst_layer.get_workspace()}});
  }
};

}  // namespace dil

#endif