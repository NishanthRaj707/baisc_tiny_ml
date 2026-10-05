#include <freertos/FreeRTOS.h>
#include <stdio.h>
#include <freertos/task.h>
#include <esp_log.h>
#include <esp_timer.h>


//Tensorflow Lite
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

static const char *tag = "bavaai";

//Allocate memory for the model
constexpr int kTensorArenaSize = (4 * 1024);
static uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main(void){
    ESP_LOGI(tag, "Starting TinyML Model...");
    
    //Load the model from Flash
    const tflite::Model* model = tflite::GetModel(g_model);
    if(model->version()!=TFLITE_SCHEMA_VERSION){
        ESP_LOGE(tag,"Error: Model version mismatch");
        return;
    }

    //Add ops to the resolver
    tflite::MicroMutableOpResolver<1> resolver;
    if(resolver.AddFullyConnected() != kTfLiteOk)
    {
        ESP_LOGE(tag,"Error: Failed to add FullyConnected op");
        return;
    }
    
    //Instantiate the interpreter
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    if(interpreter.AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(tag,"Error: Failed to allocate tensors");
        return;
    }

    // 5. Get pointers to the input and output memory addresses
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    ESP_LOGI(tag, "Interpreter ready. Arena used: %d bytes", interpreter.arena_used_bytes());
    ESP_LOGI(tag, "Model expects INT8 inputs. Training equation: y = 3x + 1");

    // 6. Execution Loop (Testing inputs x=1.0, 2.0, 3.0)
    float test_inputs[] = {1.0f, 2.0f, 3.0f, 5.0f, 10.0f};

    for (int i = 0; i < 5; i++) {
        float x_raw = test_inputs[i];

        // A. QUANTIZE: Float -> INT8
        // Formula: q = (value / scale) + zero_point
        int8_t x_quant = (int8_t)((x_raw / input->params.scale) + input->params.zero_point);
        input->data.int8[0] = x_quant;

        // B. INFER: Execute the math on the ESP32
        int64_t start_time = esp_timer_get_time();
        TfLiteStatus invoke_status = interpreter.Invoke();
        int64_t latency = esp_timer_get_time() - start_time;

        if (invoke_status != kTfLiteOk) {
            ESP_LOGE(tag, "Invoke failed on x=%.1f", x_raw);
            continue;
        }

        // C. DEQUANTIZE: INT8 -> Float
        // Formula: value = (q - zero_point) * scale
        int8_t y_quant = output->data.int8[0];
        float y_pred = (y_quant - output->params.zero_point) * output->params.scale;
        float y_true = (3.0f * x_raw) + 1.0f;

        ESP_LOGI(tag, "Input X: %5.1f | Pred Y: %6.2f (True: %6.2f) | Latency: %lld us", 
                 x_raw, y_pred, y_true, latency);

        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    ESP_LOGI(tag, "Inference completed.");
    
}




