#include <jni.h>
#include <string>
#include <mace/public/mace.h>
#include <fstream>

using namespace mace;
using namespace std;

#define USE_GPU 1

#include <chrono>
#include <android/bitmap.h>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_resnetm_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

string string_from_jni(JNIEnv *env, jstring& s)
{
    jboolean isCopy;
    const char *convertedValue = (env)->GetStringUTFChars(s, &isCopy);
    std::string r = std::string(convertedValue);
    env->ReleaseStringUTFChars(s, convertedValue);
    return r;
}

class ModelData
{
public:
  std::shared_ptr<mace::MaceEngine> engine;
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_resnetm_MainActivity_loadModel(JNIEnv *env, jobject thiz, jstring cache_dir,
                                                jstring model_data, jstring model_pb) {
    string files_dir_ = string_from_jni(env, cache_dir);
    string yoga_classifier_data_ = string_from_jni(env, model_data);
    string yoga_classifier_pb_ = string_from_jni(env, model_pb);
    const std::string storage_path = files_dir_;

#if USE_GPU
    DeviceType device_type = DeviceType::GPU;
    MaceStatus status;
    MaceEngineConfig config(device_type);
    std::shared_ptr<GPUContext> gpu_context;
    gpu_context = GPUContextBuilder()
            .SetStoragePath(storage_path)
            .Finalize();
    config.SetGPUContext(gpu_context);
    config.SetGPUHints(
            static_cast<GPUPerfHint>(GPUPerfHint::PERF_NORMAL),
            static_cast<GPUPriorityHint>(GPUPriorityHint::PRIORITY_LOW));
#else
    DeviceType device_type = DeviceType::CPU;
    MaceStatus status;
    MaceEngineConfig config(device_type);
#endif

    std::vector<std::string> input_names = {"input"};
    std::vector<std::string> output_names = {"output"};
    std::ifstream yoga_classifier_data_in( yoga_classifier_data_, std::ios::binary );
    std::vector<unsigned char> yoga_classifier_data_buffer((std::istreambuf_iterator<char>(yoga_classifier_data_in)),  (std::istreambuf_iterator<char>( )));
    std::ifstream yoga_classifier_pb_in( yoga_classifier_pb_, std::ios::binary );
    std::vector<unsigned char> yoga_classifier_pb_buffer((std::istreambuf_iterator<char>(yoga_classifier_pb_in)),  (std::istreambuf_iterator<char>( )));

    size_t data_size = yoga_classifier_data_buffer.size();
    size_t pb_size = yoga_classifier_pb_buffer.size();
    std::shared_ptr<mace::MaceEngine> engine;

    MaceStatus create_engine_status =
      CreateMaceEngineFromProto(&yoga_classifier_pb_buffer[0],
                                pb_size,
                                &yoga_classifier_data_buffer[0],
                                data_size,
                                input_names,
                                output_names,
                                config,
                                &engine);
    if (create_engine_status != MaceStatus::MACE_SUCCESS) {
        return 0;
    }
    ModelData *result = new ModelData();
    result->engine = engine;
    return long(result);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_resnetm_MainActivity_classifyImage(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                    jobject bitmap) {

    ModelData *modelData = (ModelData*)model_ptr;

    int ret;
    AndroidBitmapInfo info;
    void* pixels = 0;

    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        return -1;
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ) {
        return -2;
    }

    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &pixels)) < 0) {
        return -3;
    }

    if ((info.width != 256)||(info.height != 256))
    {
        return -10;
    }

    static const float mean[] = {0.485, 0.456, 0.406};
    static const float std[] = {0.229, 0.224, 0.225};

    auto buffer_in = std::shared_ptr<float>(new float[1*3*256*256],
                                            std::default_delete<float[]>());
    int bitmap_step = 4;
    int tensor_step = 3;
    int width = 256;
    int height = 256;
    unsigned char *src_input = reinterpret_cast<unsigned char *>(pixels);
    float *dst_input = buffer_in.get();

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < tensor_step; k++) {
                float c = src_input[i*width*bitmap_step + j*bitmap_step + k];
                c = (c / 255 - mean[k]) / std[k];
                dst_input[k*width*height + i*width + j] = c;
            }
        }
    }

    std::vector<std::string> input_names = {"input"};
    std::vector<std::string> output_names = {"output"};
    vector<vector<int64_t>> input_shapes;
    input_shapes.push_back(vector<int64_t>{1, 3, 256, 256});
    vector<vector<int64_t>> output_shapes;
    output_shapes.push_back(vector<int64_t>{1, 1000});

    std::map<std::string, mace::MaceTensor> inputs;
    inputs["input"] = mace::MaceTensor(input_shapes[0], buffer_in, DataFormat::NCHW);
    auto buffer_out = std::shared_ptr<float>(new float[1*1000],
                                             std::default_delete<float[]>());
    std::map<std::string, mace::MaceTensor> outputs;
    outputs["output"] = mace::MaceTensor(output_shapes[0], buffer_out);
    MaceStatus run_status;

    auto start = chrono::steady_clock::now();
    run_status = modelData->engine->Run(inputs, &outputs);
    auto end = chrono::steady_clock::now();

    auto code = run_status.code();
    string inference_info = run_status.information();

    float *res = outputs["output"].data().get();
    double t =  chrono::duration <double, milli> (end - start).count();

    int result = 0;
    float best_score = 0;
    for (int i = 0; i < 1000; i++)
    {
        float score = res[i];
        if (score > best_score)
        {
            best_score = score;
            result = i;
        }
    }

    AndroidBitmap_unlockPixels(env, bitmap);

    return result;
}
