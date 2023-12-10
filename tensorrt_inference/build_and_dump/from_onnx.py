import torch 
import onnx 
import tensorrt as trt 
 
 
onnx_model = 'model.onnx' 
 
device = torch.device('cuda:0') 
 
# generate ONNX model 
onnx_model = onnx.load(onnx_model) 
 
# create builder and network 
logger = trt.Logger(trt.Logger.ERROR) 
builder = trt.Builder(logger) 
EXPLICIT_BATCH = 1 << (int)( 
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
network = builder.create_network(EXPLICIT_BATCH) 
 
# parse onnx 
parser = trt.OnnxParser(network, logger) 
 
if not parser.parse(onnx_model.SerializeToString()): 
    error_msgs = '' 
    for error in range(parser.num_errors): 
        error_msgs += f'{parser.get_error(error)}\n' 
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 
 
# dump
config = builder.create_builder_config() 
config.max_workspace_size = 1<<20 
profile = builder.create_optimization_profile() 

profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224]) 
config.add_optimization_profile(profile) 
with torch.cuda.device(device): 
    engine = builder.build_engine(network, config) 
 
with open('from_onnx.engine', mode='wb') as f: 
    f.write(bytearray(engine.serialize())) 
    print("generating file done!")