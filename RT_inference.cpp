#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <utility>
#include <stdlib.h>

/////////////get class path////////////
#include <unistd.h>
#include <dirent.h>
//////////////////////////////////////

#include "NvInfer.h"
#include "NvCaffeParser.h"

//using namespace nvinfer1;
//using namespace nvcaffeparser1;

//using namespace cv;
typedef std::pair<std::string,float> mate;
#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}


// stuff we know about the network and the caffe input/output blobs
//static const int INPUT_H = 227;
//static const int INPUT_W = 227;
//static const int CHANNEL_NUM = 3;
//static const int OUTPUT_SIZE = 1000;
static const int INPUT_H = 192;
static const int INPUT_W = 192;
static const int CHANNEL_NUM = 3;
//static const int OUTPUT_SIZE = 1498; //1498
int OUTPUT_SIZE = 1498; //1498    ********************Define by yourself*****************


const std::string Model_  = "fr_1498.caffemodel";
const std::string Deploy_ = "deploy.prototxt";
const std::string Image_  = "2.jpg";
const std::string Mean_   = "mean.binaryproto";
const std::string Label_  = "labels.txt";
const std::string Path_   = "./fr_model/";

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "softmax";
//const char* OUTPUT_BLOB_NAME = "fc11_dropout";
//=================================================

//float prob[OUTPUT_SIZE];
float *prob=(float*)malloc(OUTPUT_SIZE*sizeof(float));

//================================================
// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;


std::string locateFile(const std::string& input)
{
	std::string file = Path_ + input;
	struct stat info;
	int i, MAX_DEPTH = 1;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

	return file;
}


void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 nvinfer1::IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
															  locateFile(modelFile).c_str(),
															  *network,
															  nvinfer1::DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	//builder->setHalf2Mode(true);

	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	//////(TensorRT1.0)// engine->serialize(gieModelStream);
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	nvcaffeparser1::shutdownProtobufLibrary();
}


void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize)
{
	const nvinfer1::ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * CHANNEL_NUM * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * CHANNEL_NUM * sizeof(float), cudaMemcpyHostToDevice, stream));

	context.enqueue(batchSize, buffers, stream, nullptr);

	//CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));

	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of p. */
static std::vector<int> Argmax(const float *p, int N) {
  	std::vector<std::pair<float, int> > pairs;
  	for (size_t i = 0; i < OUTPUT_SIZE; ++i)
    		pairs.push_back(std::make_pair(p[i], i));
  	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  	std::vector<int> result;
  	for (int i = 0; i < N; ++i)
    		result.push_back(pairs[i].second);
  	return result;
}

void preprocess(cv::Mat& img, cv::Size input_geometry_)
{
	cv::Mat sample, sample_resized;
	input_geometry_ = cv::Size(INPUT_W, INPUT_H);

	if (img.channels() == 3 && CHANNEL_NUM == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && CHANNEL_NUM == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && CHANNEL_NUM == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && CHANNEL_NUM == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	if (sample.size() != input_geometry_)
	    cv::resize(sample, sample_resized, input_geometry_);

	else
	    sample_resized = sample;

	img = sample_resized;
}



int main(int argc, char** argv)
{
	clock_t t1, t2, t3, t4, t5, t6, t7;

//===========================================================================================
//===========================================================================================
	
t1=clock();
	// create a GIE model from the caffe model and serialize it to a stream
    	nvinfer1::IHostMemory *gieModelStream{nullptr};
	caffeToGIEModel(Deploy_, Model_, std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);

t2=clock();
	// deserialize the engine 
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
   	if (gieModelStream) gieModelStream->destroy();

	nvinfer1::IExecutionContext *context = engine->createExecutionContext();
	std::cout<<"engine builded!!!!"<< std::endl;

t3=clock();	
	// parse the mean file 
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
	nvcaffeparser1::IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile(Mean_).c_str());
	parser->destroy();

	const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());
	meanBlob->destroy();


t4=clock();

	cv::Mat img_input = cv::imread(Path_ + Image_ , 1);
	//float prob[OUTPUT_SIZE];
	if (!img_input.empty())
	{
	
		// ready all class folder, extract image features, create feature file, label file and check file 
		cv::Mat sample,Img;
		cv::Size input_geometry_;
		input_geometry_ = cv::Size(INPUT_W, INPUT_H);
		preprocess(img_input,input_geometry_);
		Img = img_input;
		cv::Mat channel[CHANNEL_NUM];
		cv::split(Img,channel);	
		unsigned int fileData[INPUT_H*INPUT_W*CHANNEL_NUM];
		int num_time=0; 
		for(int k=0;k<CHANNEL_NUM;k++)
		{	
			for(int i=0;i<INPUT_H;i++)
			{
				for(int j=0;j<INPUT_W;j++)
				{
					fileData[num_time]=(int)channel[k].at<uchar>(i,j);
					num_time++;			
				}
			}
		}
		float data[INPUT_H*INPUT_W*CHANNEL_NUM];
		for (int i = 0; i < INPUT_H*INPUT_W*CHANNEL_NUM; i++)
		{	
			data[i] = float(fileData[i])-meanData[i];								
		}
			
		doInference(*context, data, prob, 1);
	}

t5=clock();

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	free(prob);
////============================================================================================
////============================================================================================
////============================================================================================

	//load labels.txt data
	std::vector<std::string> labels_;
	std::ifstream labels(Path_ + Label_);
	std::string line;
	while (std::getline(labels, line))
		labels_.push_back(std::string(line));	


	std::vector<int>maxN=Argmax(prob,5);  // find top 5 sort
	std::vector<mate> predictions;        //typedef std::pair<std::string,float> mate;
	for(int i=0;i<5;i++)
	{
		int idx=maxN[i];
		predictions.push_back(std::make_pair(labels_[idx],prob[idx]));
	}
	
	// Print the top N predictions. 
	for (size_t i = 0; i < predictions.size(); ++i) {
		mate p = predictions[i];
		std::cout << std::fixed << p.second << " - \""
		<< p.first << "\"" << std::endl;
	}

////============================================================================================
////============================================================================================
////============================================================================================

t6=clock();

	std::cout<<"t2-t1 time:"<<(double)(t2-t1)/(CLOCKS_PER_SEC)<<"s"<<" (create GIE model) "<<std::endl;
	std::cout<<"t3-t2 time:"<<(double)(t3-t2)/(CLOCKS_PER_SEC)<<"s"<<" (build engine "<<std::endl;
	std::cout<<"t4-t3 time:"<<(double)(t4-t3)/(CLOCKS_PER_SEC)<<"s"<<" (parse mean file) "<<std::endl;
	std::cout<<"t6-t5 time:"<<(double)(t5-t4)/(CLOCKS_PER_SEC)<<"s"<<" (doInference) "<<std::endl;
	std::cout<<"t7-t6 time:"<<(double)(t6-t5)/(CLOCKS_PER_SEC)<<"s"<<" (find top 5) "<<std::endl;
	std::cout<<"t6-t1 time:"<<(double)(t6-t1)/(CLOCKS_PER_SEC)<<"s"<<std::endl;
	return 0;
}









