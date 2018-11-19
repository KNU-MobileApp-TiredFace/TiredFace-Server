import retrain, predict, config
from subprocess import call

def execute_Train(basePath, tensor_Name):
    print('## TRAINING START')
    print('# base path:', basePath)

    _store_path = basePath + '/model/bottleneck'
    _iteration = config.ITERATION
    _model_dir = basePath + '/model/inception'
    _output_graph = basePath + '/model/retrained_graph.pb'
    _output_label = basePath + '/model/retrained_labels.txt'
    _image_dir = basePath + '/images'
    _summaries_dir = basePath + '/model/log'
    _tensor_name = tensor_Name
    cmd = "python ./retrain.py --image_dir={image_dir} \
                                    --saved_model_dir={model_dir} \
                                    --bottleneck_dir={store_path} \
                                    --how_many_training_steps={iteration} \
                                    --output_graph={output_graph} \
                                    --output_labels={output_label} \
                                    --summaries_dir={summaries_dir} \
                                    --final_tensor_name={tensor_name}".format(image_dir=_image_dir,
                                                                            model_dir=_model_dir,
                                                                            store_path=_store_path,
                                                                            iteration=_iteration,
                                                                            output_graph=_output_graph,
                                                                            output_label=_output_label,
                                                                            summaries_dir=_summaries_dir,
                                                                            tensor_name=_tensor_name)
    print('process call:', cmd)
    cmd_args = cmd.split()
    call(cmd_args)

def execute_Predict(base_Path, tensor_Name):
    modelFullPath = base_Path + '/model/retrained_graph.pb'
    labelsFullPath = base_Path + '/model/retrained_labels.txt'
    imageDir = base_Path + '/predict'
    result = predict.run_inference_on_image(modelFullPath,labelsFullPath,imageDir,tensor_Name)
    print(result)

if __name__ == '__main__':
    base_Path = config.BASE
    tensor_Name = config.TENSOR_NAME
    ds_store = "rm ./predict/.DS_Store"
    ds_store_args = ds_store.split()
    call(ds_store_args)
    #execute_Train(base_Path, tensor_Name)
    execute_Predict(base_Path, tensor_Name)